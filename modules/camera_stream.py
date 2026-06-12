"""
camera_stream.py
─────────────────────────────────────────────────────────────
Thread-safe RTSP / IP camera stream manager.

Features:
  • Automatic reconnection on failure
  • Per-camera frame buffer (drops oldest frame when full)
  • Real-time FPS measurement
  • Frame-skip support for CPU-limited deployments
  • MultiCameraManager for fleet management
"""

from __future__ import annotations

import queue
import threading
import time
import logging
import re
import os
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from typing import Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """A single captured video frame with metadata."""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    frame_number: int
    motion_detected: bool = False
    motion_area: float = 0.0


# ─────────────────────────────────────────────────────────────
# Single camera stream
# ─────────────────────────────────────────────────────────────

class CameraStream:
    """
    Wraps an OpenCV VideoCapture in a background thread.
    Supports RTSP, HTTP MJPEG and local webcam sources.
    """

    def __init__(
        self,
        camera_id: str,
        source,                     # str (URL) or int (webcam index)
        fps: int = 25,
        buffer_size: int = 2,
        reconnect_delay: int = 5,
        frame_skip: int = 1,        # process every Nth frame
        motion_detection: bool = False,
        motion_threshold: int = 500,   # minimum pixel area to count as motion
        motion_only: bool = False,     # skip frames with no motion
        resize_width: int = 0,         # resize frame width at capture (0 = no resize)
        resize_height: int = 0,        # resize frame height at capture (0 = no resize)
        rtsp_transport: str = "udp",   # udp (low-latency LAN) or tcp (reliable/WiFi)
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.target_fps = fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.frame_skip = max(1, frame_skip)
        self.motion_detection = motion_detection
        self.motion_threshold = motion_threshold
        self.motion_only = motion_only
        self.resize_width = max(0, resize_width)
        self.resize_height = max(0, resize_height)
        self.rtsp_transport = rtsp_transport  # "udp" or "tcp"

        self._frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=buffer_size)
        # Latest-frame slot: always holds the most recent captured frame.
        # Recognition reads this so it never processes a stale buffered frame.
        self._latest_frame: Optional[Frame] = None
        self._latest_lock = threading.Lock()
        self._latest_event = threading.Event()

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._frame_count = 0
        self._current_fps = 0.0
        self._fps_frame_count = 0
        self._last_fps_ts = time.time()

        # Connection health tracking
        self._connected = False
        self._reconnect_count = 0
        self._last_connected_at: Optional[float] = None
        self._last_disconnected_at: Optional[float] = None
        self._current_reconnect_delay = float(reconnect_delay)
        self._MAX_RECONNECT_DELAY = 60.0

        # Motion detection state (MOG2 background subtractor per camera)
        self._bg_subtractor: Optional[cv2.BackgroundSubtractorMOG2] = None
        self._motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ── Lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Start the background capture thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cam-{self.camera_id}",
        )
        self._thread.start()
        logger.info("[%s] Stream thread started  source=%s", self.camera_id, self.source)

    def stop(self) -> None:
        """Signal the capture thread to stop and release the capture device."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=6.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("[%s] Stream stopped", self.camera_id)

    # ── Internal helpers ────────────────────────────────────

    def _discover_stream_from_html(self, url: str) -> Optional[str]:
        """Try to discover a stream URL from an HTML page (e.g., <img src='...'>)."""
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=3) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type:
                    return None
                html = resp.read(65536).decode("utf-8", errors="ignore")
        except Exception:
            return None

        # Look for common stream-like references first.
        patterns = [
            r'src=["\']([^"\']*(?:mjpg|mjpeg|video|stream|live)[^"\']*)["\']',
            r'href=["\']([^"\']*(?:mjpg|mjpeg|video|stream|live)[^"\']*)["\']',
        ]
        for pattern in patterns:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if candidate:
                    return urljoin(url, candidate)

        return None

    def _build_network_source_candidates(self, source_url: str) -> List[str]:
        """Build candidate URLs for IP camera streams from a provided base/source URL."""
        candidates: List[str] = []

        def add(url: str) -> None:
            if url and url not in candidates:
                candidates.append(url)

        add(source_url)

        parsed = urlparse(source_url)
        if parsed.scheme in {"http", "https"}:
            base = f"{parsed.scheme}://{parsed.netloc}"
            path = parsed.path or "/"

            if path in {"", "/"}:
                for suffix in ["/video", "/mjpeg", "/stream", "/live", "/cam.mjpg"]:
                    add(base + suffix)

            discovered = self._discover_stream_from_html(source_url)
            if discovered:
                add(discovered)

        return candidates

    def _open_network_capture(self, candidate: str) -> Optional[cv2.VideoCapture]:
        """Open a network stream using the most reliable backend available."""
        if candidate.lower().startswith("rtsp://"):
            transport = self.rtsp_transport  # "udp" = low latency LAN, "tcp" = reliable/WiFi
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}"
                "|stimeout;2000000"
                "|fflags;nobuffer+discardcorrupt"
                "|flags;low_delay"
                "|max_delay;0"
                "|reorder_queue_size;0"
                "|probesize;32"
                "|analyzeduration;0"
            )
            for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
                cap = cv2.VideoCapture(candidate, backend)
                if cap is not None and cap.isOpened():
                    return cap
                if cap is not None:
                    cap.release()
            return None

        cap = cv2.VideoCapture(candidate)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        return None

    def _connect(self) -> bool:
        """Open (or re-open) the VideoCapture. Returns True on success."""
        if self._cap is not None:
            self._cap.release()

        source = self.source
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())

        # For local webcams on Windows, try explicit backends for better reliability.
        if isinstance(source, int):
            candidate_sources = []
            for idx in [source, 0, 1, 2, 3]:
                if idx not in candidate_sources:
                    candidate_sources.append(idx)

            attempts = []
            for src in candidate_sources:
                attempts.extend([
                    (src, cv2.CAP_DSHOW),
                    (src, cv2.CAP_MSMF),
                    (src, cv2.CAP_ANY),
                ])

            self._cap = None
            for src, backend in attempts:
                cap = cv2.VideoCapture(src, backend)
                if cap is not None and cap.isOpened():
                    # Validate by reading a few warm-up frames.
                    ok = False
                    for _ in range(15):
                        ret, _frame = cap.read()
                        if ret:
                            ok = True
                            break
                        time.sleep(0.03)
                    if not ok:
                        cap.release()
                        continue

                    self._cap = cap
                    self.source = src
                    logger.info(
                        "[%s] Connected with backend=%s source=%s",
                        self.camera_id,
                        backend,
                        src,
                    )
                    break
                if cap is not None:
                    cap.release()
        else:
            self._cap = None
            candidates = self._build_network_source_candidates(str(source))
            for candidate in candidates:
                logger.info("[%s] Trying source=%s", self.camera_id, candidate)
                cap = self._open_network_capture(candidate)
                if cap is None:
                    continue

                # Validate by reading warm-up frames so plain HTML URLs are rejected.
                ok = False
                for _ in range(20):
                    ret, _frame = cap.read()
                    if ret:
                        ok = True
                        break
                    time.sleep(0.03)

                if not ok:
                    cap.release()
                    continue

                self._cap = cap
                source = candidate
                self.source = candidate
                logger.info("[%s] Connected source=%s", self.camera_id, candidate)
                break

        # RTSP-specific tuning: small buffer, prefer TCP transport
        if self._cap is not None and isinstance(source, str) and source.lower().startswith("rtsp"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if self._cap is None or not self._cap.isOpened():
            if isinstance(source, str):
                tried = self._build_network_source_candidates(str(source))
                logger.error("[%s] Cannot open source: %s (tried=%s)", self.camera_id, source, tried)
            else:
                logger.error("[%s] Cannot open source: %s", self.camera_id, source)
            return False

        logger.info("[%s] Connected", self.camera_id)
        return True

    def _check_motion(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return (motion_detected, total_motion_area) using MOG2 background subtraction."""
        if self._bg_subtractor is None:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._motion_kernel)
        fg_mask = cv2.dilate(fg_mask, self._motion_kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
        return area >= self.motion_threshold, area

    def _read_live_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return the most recent frame from the camera with zero buffer lag.

        grab() reads compressed data without decoding — it is nearly instant
        when frames are already buffered inside FFmpeg.  We keep grabbing until
        grab() blocks for > 5 ms (meaning it had to wait for the network), which
        tells us the buffer is empty and we are now truly live.  We then decode
        only that last grabbed frame with retrieve().
        """
        if self._cap is None:
            return False, None
        if not self._cap.grab():
            return False, None
        # Drain any stale frames from FFmpeg's internal buffer.
        for _ in range(60):
            t0 = time.monotonic()
            if not self._cap.grab():
                break
            if (time.monotonic() - t0) * 1000 > 5.0:
                # grab() waited for the network — buffer is drained, we are live
                break
        return self._cap.retrieve()

    def _capture_loop(self) -> None:
        """Main loop: connect → read → buffer frames → reconnect on error."""
        while self._running:
            if not self._connect():
                self._connected = False
                self._last_disconnected_at = time.time()
                logger.warning(
                    "[%s] Retrying in %.0fs… (attempt #%d)",
                    self.camera_id, self._current_reconnect_delay, self._reconnect_count + 1,
                )
                time.sleep(self._current_reconnect_delay)
                # Exponential backoff: double delay each failure up to max
                self._current_reconnect_delay = min(
                    self._current_reconnect_delay * 2, self._MAX_RECONNECT_DELAY
                )
                self._reconnect_count += 1
                continue

            if self._cap is None:
                logger.error("[%s] Connection succeeded but _cap is None", self.camera_id)
                time.sleep(self._current_reconnect_delay)
                continue

            # Successful connection — reset backoff and update health state
            self._connected = True
            self._last_connected_at = time.time()
            self._current_reconnect_delay = float(self.reconnect_delay)
            if self._reconnect_count > 0:
                logger.info("[%s] Reconnected after %d attempt(s)", self.camera_id, self._reconnect_count)

            failures = 0
            while self._running:
                ret, raw = self._read_live_frame()

                if not ret or raw is None:
                    failures += 1
                    if failures >= 5:
                        self._connected = False
                        self._last_disconnected_at = time.time()
                        logger.warning("[%s] Stream lost – reconnecting", self.camera_id)
                        break
                    continue

                failures = 0
                self._frame_count += 1

                # Frame-skip: discard frames we don't need
                if self._frame_count % self.frame_skip != 0:
                    continue

                # Rolling FPS calculation
                self._fps_frame_count += 1
                elapsed = time.time() - self._last_fps_ts
                if elapsed >= 1.0:
                    self._current_fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._last_fps_ts = time.time()

                # Resize at capture time so all consumers (detection + streaming) benefit
                if self.resize_width and self.resize_height:
                    # Exact resolution (e.g. 1280×720)
                    if raw.shape[1] != self.resize_width or raw.shape[0] != self.resize_height:
                        raw = cv2.resize(
                            raw,
                            (self.resize_width, self.resize_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                elif self.resize_width and raw.shape[1] > self.resize_width:
                    # Width-only — maintain aspect ratio
                    scale = self.resize_width / raw.shape[1]
                    raw = cv2.resize(
                        raw,
                        (self.resize_width, int(raw.shape[0] * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                motion_detected = False
                motion_area = 0.0
                if self.motion_detection:
                    motion_detected, motion_area = self._check_motion(raw)
                    if self.motion_only and not motion_detected:
                        continue

                frame_obj = Frame(
                    camera_id=self.camera_id,
                    frame=raw.copy(),
                    timestamp=time.time(),
                    frame_number=self._frame_count,
                    motion_detected=motion_detected,
                    motion_area=motion_area,
                )

                # Always update the latest-frame slot so recognition
                # never reads a stale buffered frame.
                with self._latest_lock:
                    self._latest_frame = frame_obj
                self._latest_event.set()

                # Non-blocking enqueue: drop oldest frame when buffer is full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._frame_queue.put_nowait(frame_obj)
                except queue.Full:
                    pass

    # ── Public API ──────────────────────────────────────────

    def read(self, timeout: float = 0.1) -> Optional[Frame]:
        """Block up to *timeout* seconds and return the next buffered frame."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_latest(self, timeout: float = 0.15) -> Optional[Frame]:
        """Return the most recently captured frame.

        Unlike read(), this never returns a stale buffered frame — it always
        gives the newest frame the capture thread has written. Use this for
        the recognition pipeline to prevent lag accumulation over time.
        """
        signalled = self._latest_event.wait(timeout=timeout)
        if not signalled:
            return None
        self._latest_event.clear()
        with self._latest_lock:
            return self._latest_frame

    @property
    def fps(self) -> float:
        return round(self._current_fps, 1)

    @property
    def is_running(self) -> bool:
        return self._running and bool(self._thread and self._thread.is_alive())

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_stats(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "source": str(self.source),
            "fps": self.fps,
            "total_frames": self._frame_count,
            "queue_size": self._frame_queue.qsize(),
            "running": self.is_running,
            "connected": self._connected,
            "reconnect_count": self._reconnect_count,
            "last_connected_at": self._last_connected_at,
            "last_disconnected_at": self._last_disconnected_at,
            "status": "connected" if self._connected else (
                "reconnecting" if self._running else "stopped"
            ),
            "resize_width": self.resize_width,
            "resize_height": self.resize_height,
        }


# ─────────────────────────────────────────────────────────────
# Multi-camera manager
# ─────────────────────────────────────────────────────────────

class MultiCameraManager:
    """Registry and lifecycle manager for multiple CameraStream instances."""

    def __init__(self) -> None:
        self._cameras: Dict[str, CameraStream] = {}

    # ── Registration ────────────────────────────────────────

    def add_camera(
        self,
        camera_id: str,
        source,
        fps: int = 25,
        **kwargs,
    ) -> CameraStream:
        """Create and register a new camera stream (not yet started)."""
        stream = CameraStream(camera_id, source, fps, **kwargs)
        self._cameras[camera_id] = stream
        logger.info("Camera registered: %s", camera_id)
        return stream

    def add_cameras_from_config(self, cameras_cfg: List[Dict]) -> None:
        """Bulk-register cameras from the YAML config list."""
        for cam in cameras_cfg:
            self.add_camera(
                camera_id=cam["id"],
                source=cam["source"],
                fps=cam.get("fps", 25),
                motion_detection=cam.get("motion_detection", False),
                motion_threshold=cam.get("motion_threshold", 500),
                motion_only=cam.get("motion_only", False),
                resize_width=cam.get("resize_width", 0),
                resize_height=cam.get("resize_height", 0),
                rtsp_transport=cam.get("rtsp_transport", "udp"),
            )

    # ── Lifecycle ───────────────────────────────────────────

    def start_all(self) -> None:
        for stream in self._cameras.values():
            stream.start()
        logger.info("All cameras started (%d total)", len(self._cameras))

    def stop_all(self) -> None:
        for stream in self._cameras.values():
            stream.stop()
        logger.info("All cameras stopped")

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and unregister a camera. Returns False if not found."""
        cam = self._cameras.pop(camera_id, None)
        if cam is None:
            return False
        cam.stop()
        logger.info("Camera removed: %s", camera_id)
        return True

    # ── Frame access ────────────────────────────────────────

    def get_camera(self, camera_id: str) -> Optional[CameraStream]:
        return self._cameras.get(camera_id)

    def read_all(self, timeout: float = 0.05) -> List[Optional[Frame]]:
        """Return one frame from every registered camera (None if none available)."""
        return [cam.read(timeout=timeout) for cam in self._cameras.values()]

    # ── Stats ───────────────────────────────────────────────

    def get_all_stats(self) -> Dict:
        return {cid: cam.get_stats() for cid, cam in self._cameras.items()}

    @property
    def camera_ids(self) -> List[str]:
        return list(self._cameras.keys())

    def __len__(self) -> int:
        return len(self._cameras)
