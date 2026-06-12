"""
main.py
─────────────────────────────────────────────────────────────
Entry point for the real-time employee face recognition system.

Starts:
  1. Database + FAISS index
  2. Camera stream threads
  3. Processing thread pool (detect → recognise → log)
  4. FastAPI server (REST + WebSocket + MJPEG)

Usage:
  python main.py                               # uses config/config.yaml
  python main.py --config path/to/config.yaml
  python main.py --no-display                  # headless mode
  python main.py --camera-source 0             # override with webcam
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
import yaml
from dotenv import load_dotenv
from loguru import logger

# ── Load environment variables ───────────────────────────────
load_dotenv()

# ── Project imports ──────────────────────────────────────────
from modules.camera_stream import MultiCameraManager, Frame
from modules.employee_database import EmployeeDatabase
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer, FaceTracker
from modules.logging_service import LoggingService
from modules.api_server import create_app
from modules.attendance_exporter import AttendanceExporter
from modules.email_service import EmailService
from modules.door_controller import DoorController
from modules.greeting_service import GreetingService
from modules.alarm_service import AlarmService
from modules.hikvision_sdk import HikvisionSDK
from modules.sdk_event_listener import SDKEventListener


# ─────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────

def configure_logging(cfg: Dict) -> None:
    log_cfg = cfg.get("logging", {})
    level   = log_cfg.get("level", "INFO")
    log_file = log_cfg.get("file", "logs/recognition.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler and reconfigure
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(
        log_file,
        level=level,
        rotation=f"{log_cfg.get('max_size_mb', 100)} MB",
        retention=log_cfg.get("backup_count", 5),
        compression="zip",
    )

    # Bridge stdlib logging to loguru
    class _LoguruHandler(logging.Handler):
        def emit(self, record):
            try:
                lvl = logger.level(record.levelname).name
            except ValueError:
                lvl = "INFO"
            logger.opt(depth=6, exception=record.exc_info).log(lvl, record.getMessage())

    logging.basicConfig(handlers=[_LoguruHandler()], level=0, force=True)


# ─────────────────────────────────────────────────────────────
# Core recognition pipeline
# ─────────────────────────────────────────────────────────────

class RecognitionPipeline:
    """
    Pulls frames from all cameras, runs detection+recognition,
    logs results, and optionally displays annotated output.
    """

    def __init__(
        self,
        cfg: Dict,
        camera_manager: MultiCameraManager,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        tracker: FaceTracker,
        logging_svc: LoggingService,
        db: EmployeeDatabase,
        door_controller: Optional["DoorController"] = None,
        greeting_service: Optional["GreetingService"] = None,
    ) -> None:
        self.cfg = cfg
        self.camera_manager = camera_manager
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.logging_svc = logging_svc
        self.db = db
        self.door_controller = door_controller
        self.greeting_service = greeting_service

        perf = cfg.get("performance", {})
        self.display = cfg.get("performance", {}).get("display_results", True)
        self.num_threads = perf.get("num_threads", 2)
        self.batch_size  = perf.get("batch_size", 2)
        self.attendance_threshold = float(
            cfg.get("attendance", {}).get("confidence_threshold", 0.60)
        )
        self.shift_start: Optional[str] = cfg.get("attendance", {}).get("shift_start")
        self.shift_end:   Optional[str] = cfg.get("attendance", {}).get("shift_end")

        self._running = False
        self._pool: Optional[ThreadPoolExecutor] = None
        self._cam_threads: Dict[str, threading.Thread] = {}
        self._windows: dict = {}

    def start(self) -> None:
        self._running = True
        self._pool = ThreadPoolExecutor(
            max_workers=max(self.num_threads, 16),
            thread_name_prefix="recognizer",
        )
        for cam_id in self.camera_manager.camera_ids:
            self._start_camera_thread(cam_id)
        logger.info("Recognition pipeline started (cameras={})", len(self._cam_threads))

    def add_camera(self, cam_id: str) -> None:
        """Start a recognition thread for a camera added at runtime."""
        if self._running and cam_id not in self._cam_threads:
            self._start_camera_thread(cam_id)
            logger.info("Recognition thread added for camera: {}", cam_id)

    def _start_camera_thread(self, cam_id: str) -> None:
        if self._pool is None:
            return
        future = self._pool.submit(self._process_camera, cam_id)
        self._cam_threads[cam_id] = future  # type: ignore[assignment]

    def stop(self) -> None:
        self._running = False
        if self._pool:
            self._pool.shutdown(wait=False)
        if self.display:
            cv2.destroyAllWindows()
        logger.info("Recognition pipeline stopped")

    # ── Per-camera processing loop ───────────────────────────

    def _process_camera(self, cam_id: str) -> None:
        cam = self.camera_manager.get_camera(cam_id)
        if cam is None:
            logger.error("Camera not found: {}", cam_id)
            return

        logger.info("Processing loop started for camera: {}", cam_id)
        while self._running:
            # read_latest() always returns the newest frame, so slow
            # processing never causes the pipeline to fall further behind.
            frame_obj = cam.read_latest(timeout=0.15)
            if frame_obj is None:
                continue

            try:
                self._process_frame(frame_obj)
            except Exception as exc:
                logger.exception("Frame processing error on {}: {}", cam_id, exc)

    def _process_frame(self, frame_obj: Frame) -> None:
        cam_id = frame_obj.camera_id
        frame  = frame_obj.frame

        # ── Detection + embedding ───────────────────────────
        faces = self.detector.detect(frame)

        labels: List[str] = []
        colors: List = []

        for face in faces:
            if face.embedding is None:
                labels.append("No embedding")
                colors.append((100, 100, 100))
                continue

            # ── Recognition ─────────────────────────────────
            result = self.recognizer.recognize(face.embedding)
            result.bbox = face.bbox

            # ── Tracking debounce ────────────────────────────
            cx, cy = face.center
            if self.tracker.should_log(result, (cx, cy)):
                self.logging_svc.log_detection(
                    camera_id=cam_id,
                    employee_id=result.employee_id if result.is_known else None,
                    employee_name=result.name,
                    confidence=result.confidence,
                    is_known=result.is_known,
                    bbox=list(face.bbox),
                    frame=frame if self.cfg["logging"].get("log_unknown_frames") and not result.is_known else None,
                )

            # ── Attendance marking ───────────────────────────
            if result.is_known and result.confidence >= self.attendance_threshold:
                try:
                    is_new = self.db.mark_attendance(
                        employee_id=result.employee_id,
                        employee_name=result.name,
                        camera_id=cam_id,
                        confidence=result.confidence,
                        department=result.department,
                        shift_start=self.shift_start,
                        shift_end=self.shift_end,
                    )
                    if is_new:
                        logger.info(
                            "Attendance: {} ({}) — confidence {:.1%}",
                            result.name, result.employee_id, result.confidence,
                        )
                except Exception as exc:
                    logger.warning("Attendance mark failed for {}: {}", result.employee_id, exc)

            # ── Door unlock + voice greeting ─────────────────
            if result.is_known:
                if self.door_controller:
                    self.door_controller.trigger(
                        employee_id=result.employee_id,
                        employee_name=result.name,
                        camera_id=cam_id,
                        confidence=result.confidence,
                    )
                if self.greeting_service:
                    self.greeting_service.greet(
                        employee_id=result.employee_id,
                        employee_name=result.name,
                    )

            # ── Build annotation label ───────────────────────
            if result.is_known:
                label = f"{result.name}  {result.confidence:.0%}"
                colors.append((0, 220, 80))
            else:
                label = f"Unknown  {result.confidence:.0%}"
                colors.append((0, 60, 220))
            labels.append(label)

        # ── Optional display ─────────────────────────────────
        if self.display:
            annotated = frame.copy()
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox
                color = colors[i] if i < len(colors) else (100, 100, 100)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                if i < len(labels):
                    cv2.putText(
                        annotated, labels[i],
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA,
                    )

            # FPS overlay
            fps_txt = f"{cam_id}  {frame_obj.frame_number}"
            cv2.putText(annotated, fps_txt, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.imshow(f"FRS – {cam_id}", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False


# ─────────────────────────────────────────────────────────────
# Application bootstrap
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_db_url(cfg: Dict) -> str:
    db_cfg = cfg.get("database", {})
    db_type = db_cfg.get("type", "sqlite")
    if db_type == "postgresql":
        pg = db_cfg.get("postgresql", {})
        pw = os.environ.get("DB_PASSWORD", pg.get("password", ""))
        return (
            f"postgresql+psycopg2://{pg['user']}:{pw}"
            f"@{pg['host']}:{pg['port']}/{pg['name']}"
        )
    path = db_cfg.get("sqlite", {}).get("path", "data/employees.db")
    return f"sqlite:///{path}"


def main(config_path: str, no_display: bool = False, camera_source=None) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg)

    logger.info("=" * 60)
    logger.info("Employee Face Recognition System  v{}", cfg["system"]["version"])
    logger.info("=" * 60)

    if no_display:
        cfg.setdefault("performance", {})["display_results"] = False

    # ── Database ──────────────────────────────────────────────
    db = EmployeeDatabase(build_db_url(cfg))
    db.initialize()
    emp_count = db.employee_count()
    logger.info("Database ready – {} employees registered", emp_count)

    # ── Face detector ─────────────────────────────────────────
    det_cfg  = cfg.get("detection", {})
    detector = FaceDetector(
        yunet_model=det_cfg.get("yunet_model", "data/models/face_detection_yunet_2023mar.onnx"),
        sface_model=det_cfg.get("sface_model", "data/models/face_recognition_sface_2021dec.onnx"),
        det_thresh=float(det_cfg.get("det_thresh", 0.6)),
        nms_thresh=float(det_cfg.get("nms_thresh", 0.3)),
        min_face_size=int(det_cfg.get("min_face_size", 40)),
        backend=det_cfg.get("backend", "insightface"),
        insightface_model=det_cfg.get("insightface_model", "buffalo_l"),
        insightface_root=det_cfg.get("insightface_root", "data/models/insightface"),
        insightface_det_size=tuple(det_cfg.get("insightface_det_size", [640, 640])),
    )
    detector.initialize()

    # ── FAISS recognizer ──────────────────────────────────────
    rec_cfg    = cfg.get("recognition", {})
    recognizer = FaceRecognizer(
        threshold=float(rec_cfg.get("threshold", 0.35)),
        embedding_dim=int(rec_cfg.get("embedding_dim", detector.EMBEDDING_DIM)),
    )
    all_emps = db.get_all_employees_with_embeddings()
    recognizer.build_index(all_emps)
    logger.info("FAISS index ready – {} embeddings loaded", recognizer.employee_count)

    # ── Face tracker ──────────────────────────────────────────
    track_cfg = cfg.get("tracking", {})
    tracker   = FaceTracker(
        cooldown_seconds=float(track_cfg.get("cooldown_seconds", 30)),
        max_distance=int(track_cfg.get("max_distance", 80)),
    )

    # ── Cameras ───────────────────────────────────────────────
    camera_manager = MultiCameraManager()
    perf_cfg = cfg.get("performance", {})
    frame_skip = perf_cfg.get("frame_skip", 2)

    if camera_source is not None:
        # Single-camera override (CLI --camera-source)
        try:
            src = int(camera_source)
        except (ValueError, TypeError):
            src = camera_source
        camera_manager.add_camera("cam_override", src, fps=25, frame_skip=frame_skip)
    else:
        cameras_cfg = cfg.get("cameras", [])
        if not cameras_cfg:
            logger.warning("No cameras configured in config.yaml")
        for cam in cameras_cfg:
            camera_manager.add_camera(
                camera_id=cam["id"],
                source=cam["source"],
                fps=cam.get("fps", 25),
                frame_skip=frame_skip,
                motion_detection=cam.get("motion_detection", False),
                motion_threshold=cam.get("motion_threshold", 500),
                motion_only=cam.get("motion_only", False),
                resize_width=cam.get("resize_width", 0),
                resize_height=cam.get("resize_height", 0),
                rtsp_transport=cam.get("rtsp_transport", "udp"),
            )

    # ── Hikvision SDK ─────────────────────────────────────────
    sdk = HikvisionSDK(cfg)
    sdk_ready = sdk.load() and sdk.initialize() and sdk.login()
    if not sdk_ready:
        logger.info("HikvisionSDK not active — using ISAPI / HTTP fallback")

    # ── Logging service ───────────────────────────────────────
    log_cfg = cfg.get("logging", {})
    alert_cfg = cfg.get("alerts", {})
    logging_svc = LoggingService(
        db=db,
        log_frames=log_cfg.get("log_unknown_frames", False),
        frames_dir=log_cfg.get("frames_dir", "logs/frames"),
        webhook_url=alert_cfg.get("webhook_url", ""),
        unknown_alert=alert_cfg.get("unknown_person", True),
    )
    logging_svc.start()

    # ── Alarm service (unknown person alert) ──────────────────
    alarm_svc = AlarmService(cfg, sdk=sdk)
    alarm_svc.start()
    logging_svc.subscribe(alarm_svc.on_detection_event)

    # ── Attendance exporter (daily CSV + Excel at 11:59 PM) ───
    att_cfg    = cfg.get("attendance", {})
    email_svc  = EmailService(cfg)
    exporter   = AttendanceExporter(
        db=db,
        reports_dir=att_cfg.get("reports_dir", "reports"),
        email_service=email_svc,
    )
    exporter.start()

    # ── Greeting service (TTS) ────────────────────────────────
    greeting_svc = GreetingService(cfg, sdk=sdk)
    greeting_svc.start()

    # ── Door controller ───────────────────────────────────────
    door_controller = DoorController(cfg, sdk=sdk)
    door_controller.initialize()

    # ── SDK event listener (motion / intrusion / line-crossing) ──
    sdk_events = SDKEventListener(sdk, camera_id="frontcam1")
    sdk_events.start()

    # ── Recognition pipeline ──────────────────────────────────
    pipeline = RecognitionPipeline(
        cfg=cfg,
        camera_manager=camera_manager,
        detector=detector,
        recognizer=recognizer,
        tracker=tracker,
        logging_svc=logging_svc,
        db=db,
        door_controller=door_controller,
        greeting_service=greeting_svc,
    )

    # ── API server ────────────────────────────────────────────
    api_cfg = cfg.get("api", {})
    fastapi_app = create_app(
        db=db,
        recognizer=recognizer,
        camera_manager=camera_manager,
        logging_svc=logging_svc,
        face_detector=detector,
        config=cfg,
        attendance_exporter=exporter,
        door_controller=door_controller,
        pipeline=pipeline,
        tracker=tracker,
        alarm_svc=alarm_svc,
    )

    # ── Graceful shutdown ─────────────────────────────────────
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received – stopping…")
        pipeline.stop()
        camera_manager.stop_all()
        logging_svc.stop()
        exporter.stop()
        door_controller.cleanup()
        greeting_svc.stop()
        alarm_svc.stop()
        sdk_events.stop()
        sdk.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start cameras and pipeline in background ───────────────
    camera_manager.start_all()
    logger.info("Camera streams started – {} cameras", len(camera_manager))

    pipeline_thread = threading.Thread(target=pipeline.start, daemon=True)
    pipeline_thread.start()

    tracker_cleanup = threading.Thread(
        target=lambda: [time.sleep(120) or tracker.cleanup_stale() for _ in iter(int, 1)],
        daemon=True,
    )
    tracker_cleanup.start()

    # ── Start API server (blocking) ───────────────────────────
    logger.info(
        "API server starting on http://{}:{}",
        api_cfg.get("host", "0.0.0.0"),
        api_cfg.get("port", 8000),
    )
    logger.info(
        "Dashboard: http://localhost:{}/ ",
        api_cfg.get("port", 8000),
    )

    uvicorn.run(
        fastapi_app,
        host=api_cfg.get("host", "0.0.0.0"),
        port=int(api_cfg.get("port", 8000)),
        log_level="warning",
        access_log=False,
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time employee face recognition system."
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable local OpenCV display window (headless / server mode)",
    )
    parser.add_argument(
        "--camera-source",
        help="Override all cameras with a single source (RTSP URL or webcam index)",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        no_display=args.no_display,
        camera_source=args.camera_source,
    )
