"""
logging_service.py
─────────────────────────────────────────────────────────────
Centralised logging + alerting service.

Responsibilities:
  • Log detection events to database (via EmployeeDatabase)
  • Optionally save annotated frame images
  • Fire HTTP webhook alerts for unknown persons
  • Expose recent-event in-memory buffer for the dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Logging service
# ─────────────────────────────────────────────────────────────

class LoggingService:
    """
    Asynchronous detection logger with webhook alerting.

    Architecture
    ────────────
    • A background thread drains the event queue and persists
      entries to the database, keeping the hot recognition path
      latency-free.
    • An in-memory circular buffer (last N events) feeds the
      live dashboard.
    • Webhook triggers are sent via async httpx for non-blocking
      delivery.
    """

    MAX_BUFFER = 500     # in-memory events kept for the dashboard

    def __init__(
        self,
        db,                         # EmployeeDatabase instance
        log_frames: bool = False,
        frames_dir: str = "logs/frames",
        webhook_url: str = "",
        unknown_alert: bool = True,
    ) -> None:
        self._db = db
        self.log_frames = log_frames
        self.frames_dir = Path(frames_dir)
        self.webhook_url = webhook_url.strip()
        self.unknown_alert = unknown_alert

        if self.log_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        # In-memory recent-events ring buffer
        self._buffer: Deque[Dict] = deque(maxlen=self.MAX_BUFFER)
        self._buffer_lock = threading.Lock()

        # Background queue + worker thread
        self._queue: queue.Queue = queue.Queue(maxsize=2000)
        self._worker = threading.Thread(
            target=self._drain_loop,
            daemon=True,
            name="logging-worker",
        )

        # Subscribers for real-time push (dashboard WebSocket)
        self._subscribers: List[Callable[[Dict], None]] = []
        self._sub_lock = threading.Lock()

        self._running = False

    # ── Lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._worker.start()
        logger.info("LoggingService started")

    def stop(self) -> None:
        self._running = False
        self._queue.join()       # Drain remaining entries
        logger.info("LoggingService stopped")

    # ── Enqueue ─────────────────────────────────────────────

    def log_detection(
        self,
        camera_id: str,
        employee_id: Optional[str],
        employee_name: str,
        confidence: float,
        is_known: bool,
        bbox: Optional[List[int]] = None,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Non-blocking enqueue of a detection event.
        If the queue is full the event is dropped with a warning.
        """
        event = {
            "camera_id": camera_id,
            "employee_id": employee_id,
            "employee_name": employee_name,
            "confidence": round(float(confidence), 4),
            "is_known": is_known,
            "bbox": bbox or [],
            "frame": frame,    # ndarray or None – heavy, stored separately
            "timestamp": (timestamp or datetime.now()).isoformat(),
        }
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full – detection event dropped.")

    # ── Background worker ───────────────────────────────────

    def _drain_loop(self) -> None:
        while self._running or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._persist(event)
            except Exception as exc:
                logger.error("Failed to persist event: %s", exc)
            finally:
                self._queue.task_done()

    def _persist(self, event: Dict) -> None:
        frame: Optional[np.ndarray] = event.pop("frame", None)

        # Optionally save the frame image
        frame_path = ""
        if self.log_frames and frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cam = event["camera_id"]
            eid = event["employee_id"] or "unknown"
            fname = self.frames_dir / f"{cam}_{eid}_{ts}.jpg"
            try:
                cv2.imwrite(str(fname), frame)
                frame_path = str(fname)
            except Exception as exc:
                logger.warning("Could not save frame: %s", exc)

        # Write to DB
        bbox = event.get("bbox", [])
        entry = self._db.log_detection(
            camera_id=event["camera_id"],
            employee_id=event.get("employee_id"),
            employee_name=event.get("employee_name", "Unknown"),
            confidence=event["confidence"],
            is_known=event["is_known"],
            bbox=bbox,
            frame_path=frame_path,
            timestamp=datetime.fromisoformat(event["timestamp"]),
        )

        # Add to in-memory buffer
        record = {**event, "frame_path": frame_path, "id": entry.id if entry else None}
        with self._buffer_lock:
            self._buffer.appendleft(record)

        # Notify WebSocket subscribers
        self._notify_subscribers(record)

        # Fire webhook for unknown faces
        if not event["is_known"] and self.unknown_alert and self.webhook_url:
            self._send_webhook(record)

    # ── Dashboard event buffer ──────────────────────────────

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self._buffer_lock:
            # Return safe copy without raw frame data
            return [
                {k: v for k, v in e.items() if k != "frame"}
                for e in list(self._buffer)[:limit]
            ]

    # ── Real-time subscribers ───────────────────────────────

    def subscribe(self, callback: Callable[[Dict], None]) -> None:
        with self._sub_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict], None]) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify_subscribers(self, event: Dict) -> None:
        clean = {k: v for k, v in event.items() if k != "frame"}
        with self._sub_lock:
            for cb in list(self._subscribers):
                try:
                    cb(clean)
                except Exception as exc:
                    logger.debug("Subscriber callback error: %s", exc)

    # ── Webhook ─────────────────────────────────────────────

    def _send_webhook(self, event: Dict) -> None:
        """
        Fire-and-forget POST to the configured webhook URL.
        Runs in a separate thread to avoid blocking the worker.
        """
        payload = {
            "alert": "unknown_person_detected",
            "camera_id": event.get("camera_id"),
            "timestamp": event.get("timestamp"),
            "confidence": event.get("confidence"),
        }

        def _post() -> None:
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(self.webhook_url, json=payload)
                    logger.debug("Webhook sent: HTTP %d", resp.status_code)
            except Exception as exc:
                logger.warning("Webhook failed: %s", exc)

        threading.Thread(target=_post, daemon=True).start()
