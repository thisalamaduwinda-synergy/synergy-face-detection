#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_qt.py
──────────────────────────────────────────────────────────────
PyQt6 desktop application for real-time employee face recognition.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │                  Title bar + Stats row                  │
  ├──────────────┬──────────────────────────┬───────────────┤
  │   Employee   │    Live Camera Feed      │  Detection    │
  │   Roster     │   (annotated frames)     │    Log        │
  │   (left)     │       (centre)           │   (right)     │
  └──────────────┴──────────────────────────┴───────────────┘

Usage:
    python main_qt.py
    python main_qt.py --config config/config.yaml
    python main_qt.py --camera 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize,
)
from PyQt6.QtGui import (
    QImage, QPixmap, QColor, QFont, QIcon,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QFrame,
    QDialog, QLineEdit, QFormLayout,
    QFileDialog, QMessageBox,
    QSplitter, QSizePolicy, QToolButton,
    QDialogButtonBox, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
)
from PyQt6.QtCore import QDate

load_dotenv()

from modules.employee_database import EmployeeDatabase
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer, FaceTracker
from modules.camera_stream import CameraStream

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# Dark-theme colour constants
# ─────────────────────────────────────────────────────────────

C_BG          = "#0f172a"
C_PANEL       = "#1e293b"
C_CARD        = "#334155"
C_BORDER      = "#475569"
C_ACCENT      = "#6366f1"
C_ACCENT_H    = "#818cf8"
C_GREEN       = "#22c55e"
C_RED         = "#ef4444"
C_AMBER       = "#f59e0b"
C_TEXT        = "#f1f5f9"
C_MUTED       = "#94a3b8"

APP_STYLE = f"""
QMainWindow, QWidget      {{ background: {C_BG}; color: {C_TEXT};
                             font-family: 'Segoe UI', sans-serif; }}
QSplitter::handle         {{ background: {C_BORDER}; width: 2px; height: 2px; }}
QScrollArea               {{ border: none; background: transparent; }}
QScrollBar:vertical       {{ background: {C_PANEL}; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: {C_CARD}; border-radius: 3px;
                               min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLineEdit {{ background: {C_CARD}; color: {C_TEXT}; border: 1px solid {C_BORDER};
             border-radius: 6px; padding: 8px 12px; font-size: 13px; }}
QLineEdit:focus {{ border-color: {C_ACCENT}; }}
QDialog   {{ background: {C_PANEL}; color: {C_TEXT};
             font-family: 'Segoe UI', sans-serif; }}
QMessageBox {{ background: {C_PANEL}; color: {C_TEXT}; }}
"""


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> Dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame to QPixmap (no copy via buffer protocol)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy())   # .copy() detaches from the buffer


def emp_color(emp_id: str) -> str:
    """Deterministic HSL colour derived from an employee ID string."""
    hue = sum(ord(c) * (i + 1) for i, c in enumerate(emp_id)) % 360
    return f"hsl({hue}, 58%, 54%)"


def get_initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


def styled_btn(text: str, color: str = C_ACCENT,
               hover: str = C_ACCENT_H) -> QPushButton:
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {color}; color: #fff; border: none;
            border-radius: 6px; padding: 8px 18px;
            font-size: 13px; font-weight: 600;
        }}
        QPushButton:hover  {{ background: {hover}; }}
        QPushButton:pressed {{ background: {color}; opacity: 0.8; }}
        QPushButton:disabled {{ background: {C_CARD}; color: {C_MUTED}; }}
    """)
    return btn


# ─────────────────────────────────────────────────────────────
# Camera + Recognition background worker (QThread)
# ─────────────────────────────────────────────────────────────

class CameraWorker(QThread):
    """
    Runs in a background QThread:
      capture → detect → recognise → emit annotated frame + detection events

    Signals
    -------
    frame_ready(object)  – annotated BGR ndarray, ready for QLabel display
    detection(dict)      – one logged detection event
    status(str)          – status message for the status bar
    fps_update(float)    – current processing FPS
    """

    frame_ready = pyqtSignal(object)   # np.ndarray
    detection   = pyqtSignal(dict)
    status      = pyqtSignal(str)
    fps_update  = pyqtSignal(float)

    def __init__(self, cfg: Dict, db: EmployeeDatabase) -> None:
        super().__init__()
        self.cfg = cfg
        self.db  = db
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None

        det = cfg.get("detection", {})
        rec = cfg.get("recognition", {})
        trk = cfg.get("tracking", {})
        att = cfg.get("attendance", {})
        self._attendance_threshold = float(att.get("confidence_threshold", 0.60))
        self._shift_start: Optional[str] = att.get("shift_start")
        self._shift_end:   Optional[str] = att.get("shift_end")

        # Live-pipeline detector
        self.detector = FaceDetector(
            yunet_model   = det.get("yunet_model",
                                    "data/models/face_detection_yunet_2023mar.onnx"),
            sface_model   = det.get("sface_model",
                                    "data/models/face_recognition_sface_2021dec.onnx"),
            det_thresh    = float(det.get("det_thresh", 0.6)),
            nms_thresh    = float(det.get("nms_thresh", 0.3)),
            min_face_size = int(det.get("min_face_size", 40)),
            backend       = det.get("backend", "insightface"),
            insightface_model = det.get("insightface_model", "buffalo_l"),
            insightface_root = det.get("insightface_root", "data/models/insightface"),
            insightface_det_size = tuple(det.get("insightface_det_size", [640, 640])),
        )

        # Dedicated registration detector (isolated from live pipeline state)
        self.reg_detector = FaceDetector(
            yunet_model   = det.get("yunet_model",
                                    "data/models/face_detection_yunet_2023mar.onnx"),
            sface_model   = det.get("sface_model",
                                    "data/models/face_recognition_sface_2021dec.onnx"),
            det_thresh    = 0.45,
            nms_thresh    = 0.3,
            min_face_size = 20,
            backend       = det.get("backend", "insightface"),
            insightface_model = det.get("insightface_model", "buffalo_l"),
            insightface_root = det.get("insightface_root", "data/models/insightface"),
            insightface_det_size = tuple(det.get("insightface_det_size", [640, 640])),
        )

        self.recognizer = FaceRecognizer(
            threshold     = float(rec.get("threshold", 0.35)),
            embedding_dim = int(rec.get("embedding_dim", 512)),
        )
        self._recognizer_threshold = float(rec.get("threshold", 0.35))
        self.tracker = FaceTracker(
            cooldown_seconds = float(trk.get("cooldown_seconds", 30)),
        )

        cams = cfg.get("cameras") or []
        self._cam_id     = cams[0].get("id", "cam_01") if cams else None
        self._cam_source = cams[0].get("source", None) if cams else None
        self._cam: Optional[CameraStream] = None

    # ── Public API ───────────────────────────────────────────

    def rebuild_index(self) -> None:
        """Reload all employees from DB and rebuild FAISS index."""
        if self.recognizer.dim != self.detector.EMBEDDING_DIM:
            self.recognizer = FaceRecognizer(
                threshold=self._recognizer_threshold,
                embedding_dim=self.detector.EMBEDDING_DIM,
            )
        employees = self.db.get_all_employees_with_embeddings()
        self.recognizer.build_index(employees)
        self.status.emit(f"FAISS index rebuilt — {len(employees)} employee(s)")

    def get_latest_raw_frame(self) -> Optional[np.ndarray]:
        """Thread-safe access to the most recent raw camera frame."""
        with self._frame_lock:
            return self._latest_raw.copy() if self._latest_raw is not None else None

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Thread-safe access to the most recent annotated (recognition overlay) frame."""
        with self._frame_lock:
            return self._latest_annotated

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a face embedding from a still image using multi-pass fallback.
        Uses reg_detector (dedicated instance, called from the main/UI thread).
        """
        # Pass 1 – normal
        faces = self.reg_detector.detect(image)
        if faces:
            return faces[0].embedding

        # Pass 2 – 1.5× upscale with lower threshold
        h, w = image.shape[:2]
        bigger = cv2.resize(image, (int(w * 1.5), int(h * 1.5)))
        faces = self.reg_detector.detect(bigger)
        if faces:
            return faces[0].embedding

        # Pass 3 – CLAHE contrast enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        faces = self.reg_detector.detect(enhanced)
        if faces:
            return faces[0].embedding

        return None

    def stop(self) -> None:
        self._running = False

    # ── QThread.run ──────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        self.status.emit("Loading face recognition models…")

        try:
            self.detector.initialize()
            self.reg_detector.initialize()
        except Exception as exc:
            self.status.emit(f"Model load failed: {exc}")
            return

        # Keep FAISS dimensionality aligned with the active detector backend.
        if self.recognizer.dim != self.detector.EMBEDDING_DIM:
            self.recognizer = FaceRecognizer(
                threshold=self._recognizer_threshold,
                embedding_dim=self.detector.EMBEDDING_DIM,
            )

        # Build initial FAISS index
        employees = self.db.get_all_employees_with_embeddings()
        self.recognizer.build_index(employees)
        self.status.emit(f"Models ready — {len(employees)} employee(s) in index")

        # Start camera stream
        if self._cam_source is None:
            self.status.emit("No camera configured — use '+ Camera' to add one.")
            return
        self._cam = CameraStream(
            camera_id  = self._cam_id,
            source     = self._cam_source,
            fps        = 30,
            frame_skip = self.cfg.get("performance", {}).get("frame_skip", 2),
        )
        self._cam.start()
        self.status.emit("Camera started — recognising faces…")

        fps_count = 0
        fps_ts    = time.time()

        while self._running:
            frame_obj = self._cam.read(timeout=0.1)
            if frame_obj is None:
                continue

            raw = frame_obj.frame

            # Store raw frame for the registration dialog preview
            with self._frame_lock:
                self._latest_raw = raw.copy()

            # Detection + embedding
            annotated = raw.copy()
            try:
                faces = self.detector.detect(raw)
                for face in faces:
                    if face.embedding is None:
                        continue

                    result = self.recognizer.recognize(face.embedding)
                    result.bbox = face.bbox

                    cx, cy = face.center
                    if self.tracker.should_log(result, (cx, cy)):
                        # Persist to DB directly (no async needed here)
                        try:
                            self.db.log_detection(
                                camera_id     = self._cam_id,
                                employee_id   = result.employee_id if result.is_known else None,
                                employee_name = result.name,
                                confidence    = result.confidence,
                                is_known      = result.is_known,
                                bbox          = list(face.bbox),
                            )
                        except Exception:
                            pass  # never crash the recognition loop

                        # Mark attendance when confidence is high enough
                        if result.is_known and result.confidence >= self._attendance_threshold:
                            try:
                                self.db.mark_attendance(
                                    employee_id   = result.employee_id,
                                    employee_name = result.name,
                                    camera_id     = self._cam_id,
                                    confidence    = result.confidence,
                                    department    = result.department,
                                    shift_start   = self._shift_start,
                                    shift_end     = self._shift_end,
                                )
                            except Exception:
                                pass

                        self.detection.emit({
                            "timestamp":   datetime.now().strftime("%H:%M:%S"),
                            "name":        result.name,
                            "employee_id": result.employee_id,
                            "confidence":  result.confidence,
                            "is_known":    result.is_known,
                            "camera_id":   self._cam_id,
                        })

                    # Draw bounding box + label
                    x1, y1, x2, y2 = face.bbox
                    box_color = (40, 220, 80) if result.is_known else (50, 50, 220)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{result.name}  {result.confidence:.0%}"
                    cv2.rectangle(annotated, (x1, y1 - 26), (x2, y1), box_color, cv2.FILLED)
                    cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                                (255, 255, 255), 1, cv2.LINE_AA)

            except Exception as exc:
                logger.exception("Frame processing error: %s", exc)

            with self._frame_lock:
                self._latest_annotated = annotated

            # FPS counter
            fps_count += 1
            now = time.time()
            if now - fps_ts >= 1.0:
                self.fps_update.emit(fps_count / (now - fps_ts))
                fps_count = 0
                fps_ts    = now

        if self._cam:
            self._cam.stop()
        self.status.emit("Camera stopped.")


# ─────────────────────────────────────────────────────────────
# Stats Bar (top)
# ─────────────────────────────────────────────────────────────

class _StatCard(QFrame):
    def __init__(self, title: str, color: str = C_ACCENT, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            _StatCard {{
                background: {C_PANEL}; border: 1px solid {C_BORDER};
                border-left: 3px solid {color}; border-radius: 8px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self._val = QLabel("0")
        self._val.setStyleSheet(
            f"color: {color}; font-size: 26px; font-weight: 700;")
        self._ttl = QLabel(title)
        self._ttl.setStyleSheet(
            f"color: {C_MUTED}; font-size: 10px; font-weight: 600;")
        lay.addWidget(self._val)
        lay.addWidget(self._ttl)

    def set_value(self, v) -> None:
        self._val.setText(str(v))


class StatsBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self._emp     = _StatCard("Registered",       C_ACCENT)
        self._total   = _StatCard("Total Detections", C_AMBER)
        self._known   = _StatCard("Known",            C_GREEN)
        self._unknown = _StatCard("Unknown",          C_RED)
        self._fps     = _StatCard("FPS",              C_MUTED)

        for card in (self._emp, self._total, self._known, self._unknown, self._fps):
            lay.addWidget(card)

    def update_stats(self, s: Dict) -> None:
        self._emp.set_value(s.get("registered_employees", 0))
        self._total.set_value(s.get("total_detections", 0))
        self._known.set_value(s.get("known_detections", 0))
        self._unknown.set_value(s.get("unknown_detections", 0))

    def update_fps(self, fps: float) -> None:
        self._fps.set_value(f"{fps:.1f}")


# ─────────────────────────────────────────────────────────────
# Video Display (centre)
# ─────────────────────────────────────────────────────────────

class VideoDisplay(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(320, 240))
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"""
            background: #0a0f1e;
            color: {C_MUTED};
            font-size: 14px;
            border-radius: 8px;
        """)
        self.setText("◉  Waiting for camera…")

    def update_frame(self, frame: np.ndarray) -> None:
        pix = bgr_to_pixmap(frame).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(pix)
        self.setText("")


# ─────────────────────────────────────────────────────────────
# Employee Panel (left side-bar)
# ─────────────────────────────────────────────────────────────

class _EmployeeCard(QFrame):
    delete_clicked = pyqtSignal(str)
    card_clicked   = pyqtSignal(str)

    def __init__(self, emp: Dict, parent=None) -> None:
        super().__init__(parent)
        self._emp_id = emp["employee_id"]
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            _EmployeeCard {{
                background: {C_CARD}; border-radius: 8px;
                border: 1px solid {C_BORDER};
            }}
            _EmployeeCard:hover {{ border-color: {C_ACCENT}; }}
        """)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        # Coloured avatar
        color = emp_color(self._emp_id)
        av = QLabel(get_initials(emp["name"]))
        av.setFixedSize(QSize(36, 36))
        av.setAlignment(Qt.AlignmentFlag.AlignCenter)
        av.setStyleSheet(f"""
            background: {color}; border-radius: 18px;
            color: #fff; font-weight: 700; font-size: 13px;
        """)
        row.addWidget(av)

        # Name + meta
        info = QVBoxLayout()
        info.setSpacing(1)
        nm = QLabel(emp["name"])
        nm.setStyleSheet(f"color: {C_TEXT}; font-weight: 600; font-size: 13px;")
        mt = QLabel(f"{emp['employee_id']}  ·  {emp.get('department') or '—'}")
        mt.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        info.addWidget(nm)
        info.addWidget(mt)
        row.addLayout(info, stretch=1)

        # Delete button
        del_btn = QToolButton()
        del_btn.setText("✕")
        del_btn.setFixedSize(QSize(22, 22))
        del_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent; color: {C_MUTED};
                border: none; border-radius: 4px; font-size: 11px;
            }}
            QToolButton:hover {{ background: {C_RED}; color: #fff; }}
        """)
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self._emp_id))
        row.addWidget(del_btn)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.card_clicked.emit(self._emp_id)
        super().mousePressEvent(event)


class EmployeePanel(QWidget):
    add_requested    = pyqtSignal()
    delete_requested = pyqtSignal(str)
    detail_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(272)
        self.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # Header row
        hdr = QHBoxLayout()
        ttl = QLabel("Employees")
        ttl.setStyleSheet(f"color: {C_TEXT}; font-size: 15px; font-weight: 700;")
        add_btn = QPushButton("+ Add")
        add_btn.setFixedSize(QSize(62, 28))
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_ACCENT}; color: #fff; border: none;
                border-radius: 5px; font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {C_ACCENT_H}; }}
        """)
        add_btn.clicked.connect(self.add_requested)
        hdr.addWidget(ttl)
        hdr.addStretch()
        hdr.addWidget(add_btn)
        lay.addLayout(hdr)

        # Scrollable card list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._cards_lay = QVBoxLayout(self._container)
        self._cards_lay.setContentsMargins(0, 0, 4, 0)
        self._cards_lay.setSpacing(6)
        self._cards_lay.addStretch()

        scroll.setWidget(self._container)
        lay.addWidget(scroll, stretch=1)

        self._count = QLabel("0 employees")
        self._count.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        lay.addWidget(self._count)

    def refresh(self, employees: List[Dict]) -> None:
        # Remove all cards but keep the trailing stretch
        while self._cards_lay.count() > 1:
            item = self._cards_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for emp in employees:
            card = _EmployeeCard(emp)
            card.delete_clicked.connect(self.delete_requested)
            card.card_clicked.connect(self.detail_requested)
            self._cards_lay.insertWidget(
                self._cards_lay.count() - 1, card)

        n = len(employees)
        self._count.setText(f"{n} employee{'s' if n != 1 else ''}")


# ─────────────────────────────────────────────────────────────
# Detection Log Panel (right side-bar)
# ─────────────────────────────────────────────────────────────

class _LogEntry(QFrame):
    def __init__(self, event: Dict, parent=None) -> None:
        super().__init__(parent)
        color = C_GREEN if event.get("is_known") else C_RED
        self.setStyleSheet(f"""
            _LogEntry {{
                background: {C_CARD};
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
        """)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(8)

        info = QVBoxLayout()
        info.setSpacing(1)
        nm = QLabel(event.get("name", "Unknown"))
        nm.setStyleSheet(f"color: {C_TEXT}; font-weight: 600; font-size: 12px;")
        conf = event.get("confidence", 0.0)
        conf_str = f"{conf:.0%}" if isinstance(conf, float) else str(conf)
        sub = QLabel(f"{event.get('camera_id', '')}  ·  {conf_str}")
        sub.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        info.addWidget(nm)
        info.addWidget(sub)
        row.addLayout(info, stretch=1)

        ts = QLabel(event.get("timestamp", ""))
        ts.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        ts.setAlignment(Qt.AlignmentFlag.AlignRight |
                        Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(ts)


class LogPanel(QWidget):
    _MAX = 120   # maximum visible entries

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(290)
        self.setStyleSheet(f"background: {C_PANEL}; border-radius: 8px;")
        self._entry_count = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        hdr = QLabel("Detection Log")
        hdr.setStyleSheet(
            f"color: {C_TEXT}; font-size: 15px; font-weight: 700;")
        lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._log_lay = QVBoxLayout(self._container)
        self._log_lay.setContentsMargins(0, 0, 4, 0)
        self._log_lay.setSpacing(5)
        self._log_lay.addStretch()

        scroll.setWidget(self._container)
        lay.addWidget(scroll, stretch=1)

    def add_event(self, event: Dict) -> None:
        entry = _LogEntry(event)
        # Insert newest at top (position 0, before the stretch at end)
        self._log_lay.insertWidget(0, entry)
        self._entry_count += 1

        # Prune oldest entries
        while self._entry_count > self._MAX:
            # The stretch is at count()-1; oldest entries accumulate just before it
            item = self._log_lay.takeAt(self._log_lay.count() - 2)
            if item and item.widget():
                item.widget().deleteLater()
            self._entry_count -= 1


# ─────────────────────────────────────────────────────────────
# Add Employee Dialog
# ─────────────────────────────────────────────────────────────

class AddEmployeeDialog(QDialog):
    def __init__(self, worker: CameraWorker, db: EmployeeDatabase,
                 parent=None) -> None:
        super().__init__(parent)
        self.worker = worker
        self.db     = db
        self._photo: Optional[np.ndarray] = None
        self._live  = False

        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._tick_preview)

        self.setWindowTitle("Register New Employee")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(f"""
            QDialog  {{ background: {C_PANEL}; color: {C_TEXT};
                        font-family: 'Segoe UI', sans-serif; }}
            QLabel   {{ color: {C_TEXT}; font-size: 13px; }}
            QLineEdit {{ background: {C_CARD}; color: {C_TEXT};
                         border: 1px solid {C_BORDER}; border-radius: 6px;
                         padding: 8px 12px; font-size: 13px; }}
            QLineEdit:focus {{ border-color: {C_ACCENT}; }}
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(22, 22, 22, 22)

        # Title
        ttl = QLabel("Add New Employee")
        ttl.setStyleSheet(
            f"font-size: 17px; font-weight: 700; color: {C_TEXT};")
        root.addWidget(ttl)

        # Form fields
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._id_edit   = QLineEdit(); self._id_edit.setPlaceholderText("e.g. EMP042")
        self._nm_edit   = QLineEdit(); self._nm_edit.setPlaceholderText("Full name")
        self._dep_edit  = QLineEdit(); self._dep_edit.setPlaceholderText("e.g. Engineering")
        form.addRow("Employee ID :", self._id_edit)
        form.addRow("Name :",        self._nm_edit)
        form.addRow("Department :",  self._dep_edit)
        root.addLayout(form)

        # Photo preview
        sec = QLabel("Photo")
        sec.setStyleSheet(f"color: {C_MUTED}; font-size: 11px; font-weight: 600;")
        root.addWidget(sec)

        self._preview = QLabel("No photo selected")
        self._preview.setFixedSize(QSize(260, 195))
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet(
            f"background: #0a0f1e; border-radius: 8px; color: {C_MUTED};"
            f"font-size: 12px; border: 1px solid {C_BORDER};")
        root.addWidget(self._preview, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Camera / file buttons
        btn_row = QHBoxLayout()
        self._cam_btn = styled_btn("📷  Camera", C_CARD, "#475569")
        self._cap_btn = styled_btn("⏺  Capture", C_GREEN, "#16a34a")
        self._cap_btn.setEnabled(False)
        self._file_btn = styled_btn("🗂  Upload", C_CARD, "#475569")
        self._cam_btn.clicked.connect(self._toggle_camera)
        self._cap_btn.clicked.connect(self._capture)
        self._file_btn.clicked.connect(self._pick_file)
        btn_row.addWidget(self._cam_btn)
        btn_row.addWidget(self._cap_btn)
        btn_row.addWidget(self._file_btn)
        root.addLayout(btn_row)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {C_AMBER}; font-size: 12px;")
        root.addWidget(self._status)

        # Dialog buttons
        dlg_btns = QHBoxLayout()
        dlg_btns.addStretch()
        cancel = styled_btn("Cancel", C_CARD, "#475569")
        cancel.clicked.connect(self.reject)
        ok = styled_btn("Register", C_ACCENT, C_ACCENT_H)
        ok.clicked.connect(self._register)
        dlg_btns.addWidget(cancel)
        dlg_btns.addWidget(ok)
        root.addLayout(dlg_btns)

    # ── Camera preview ───────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._live:
            self._preview_timer.stop()
            self._live = False
            self._cam_btn.setText("📷  Camera")
            self._cap_btn.setEnabled(False)
        else:
            self._live = True
            self._cam_btn.setText("⏹  Stop")
            self._cap_btn.setEnabled(True)
            self._preview_timer.start(66)   # ~15 fps

    def _tick_preview(self) -> None:
        frame = self.worker.get_latest_raw_frame()
        if frame is None:
            return
        frame = cv2.flip(frame, 1)  # mirror
        pix = bgr_to_pixmap(frame).scaled(
            260, 195,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pix)

    def _capture(self) -> None:
        frame = self.worker.get_latest_raw_frame()
        if frame is None:
            self._status.setText("Camera not ready yet.")
            return
        self._photo = cv2.flip(frame, 1)
        self._preview_timer.stop()
        self._live = False
        self._cam_btn.setText("📷  Camera")
        self._cap_btn.setEnabled(False)
        pix = bgr_to_pixmap(self._photo).scaled(
            260, 195,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pix)
        self._status.setText("Photo captured ✓")

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._status.setText("Could not read image file.")
            return
        self._photo = img
        pix = bgr_to_pixmap(img).scaled(
            260, 195,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pix)
        self._status.setText("Photo loaded ✓")

    # ── Registration ─────────────────────────────────────────

    def _register(self) -> None:
        emp_id = self._id_edit.text().strip()
        name   = self._nm_edit.text().strip()
        dept   = self._dep_edit.text().strip()

        if not emp_id:
            self._status.setText("Employee ID is required.")
            return
        if not name:
            self._status.setText("Name is required.")
            return
        if self._photo is None:
            self._status.setText("Please capture or upload a photo first.")
            return

        self._status.setText("Detecting face…")
        QApplication.processEvents()

        embedding = self.worker.extract_embedding(self._photo)
        if embedding is None:
            self._status.setText(
                "No face detected in the photo. Try a clearer, front-facing image.")
            return

        save_dir = BASE_DIR / "data" / "employees"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c for c in emp_id if c.isalnum() or c in "-_") or f"emp_{uuid4().hex[:8]}"
        photo_path = save_dir / f"{safe_id}.jpg"
        if not cv2.imwrite(str(photo_path), self._photo):
            self._status.setText("Could not save employee photo.")
            return

        try:
            self.db.add_employee(emp_id, name, dept, embedding, str(photo_path))
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        except Exception as exc:
            self._status.setText(f"Database error: {exc}")
            return

        self.worker.rebuild_index()
        self.accept()

    def closeEvent(self, event) -> None:
        self._preview_timer.stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────
# Employee Detail Dialog
# ─────────────────────────────────────────────────────────────

class EmployeeDetailDialog(QDialog):
    delete_requested = pyqtSignal(str)

    def __init__(self, emp: Dict, parent=None) -> None:
        super().__init__(parent)
        self.emp = emp
        self.setWindowTitle("Employee Details")
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setStyleSheet(f"""
            QDialog  {{ background: {C_PANEL}; color: {C_TEXT};
                        font-family: 'Segoe UI', sans-serif; }}
            QLabel   {{ color: {C_TEXT}; font-size: 13px; }}
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(16)
        lay.setContentsMargins(22, 22, 22, 22)

        # Avatar + name
        hdr = QHBoxLayout()
        color = emp_color(self.emp["employee_id"])
        av = QLabel(get_initials(self.emp["name"]))
        av.setFixedSize(QSize(52, 52))
        av.setAlignment(Qt.AlignmentFlag.AlignCenter)
        av.setStyleSheet(
            f"background: {color}; border-radius: 26px;"
            f"color: #fff; font-weight: 700; font-size: 18px;")
        hdr.addWidget(av)

        hi = QVBoxLayout()
        hi.setSpacing(2)
        nm = QLabel(self.emp["name"])
        nm.setStyleSheet(
            f"font-size: 17px; font-weight: 700; color: {C_TEXT};")
        eid = QLabel(self.emp["employee_id"])
        eid.setStyleSheet(f"font-size: 12px; color: {C_MUTED};")
        hi.addWidget(nm)
        hi.addWidget(eid)
        hdr.addLayout(hi, stretch=1)
        lay.addLayout(hdr)

        # Detail grid
        grid = QGridLayout()
        grid.setSpacing(8)

        reg_at = self.emp.get("registered_at", "")
        if reg_at and len(reg_at) >= 19:
            reg_at = reg_at[:19].replace("T", "  ")

        items = [
            ("Employee ID",  self.emp.get("employee_id", "—")),
            ("Department",   self.emp.get("department") or "—"),
            ("Registered",   reg_at or "—"),
            ("Status",       "Active" if self.emp.get("is_active") else "Inactive"),
            ("Embedding",    "✓ Present" if self.emp.get("has_embedding") else "✗ Missing"),
        ]
        for idx, (lbl, val) in enumerate(items):
            row, col = divmod(idx, 2)
            box = QFrame()
            box.setStyleSheet(
                f"background: {C_CARD}; border-radius: 6px;")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(10, 8, 10, 8)
            bl.setSpacing(2)
            tl = QLabel(lbl)
            tl.setStyleSheet(f"color: {C_MUTED}; font-size: 10px; font-weight: 600;")
            vl = QLabel(val)
            vl.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 500;")
            bl.addWidget(tl)
            bl.addWidget(vl)
            grid.addWidget(box, row, col)

        lay.addLayout(grid)

        # Buttons
        br = QHBoxLayout()
        close_btn = styled_btn("Close", C_CARD, "#475569")
        close_btn.clicked.connect(self.accept)
        del_btn = styled_btn("Delete Employee", C_RED, "#dc2626")
        del_btn.clicked.connect(self._ask_delete)
        br.addWidget(close_btn)
        br.addStretch()
        br.addWidget(del_btn)
        lay.addLayout(br)

    def _ask_delete(self) -> None:
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{self.emp['name']}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(self.emp["employee_id"])
            self.accept()


# ─────────────────────────────────────────────────────────────
# Attendance Dialog
# ─────────────────────────────────────────────────────────────

class AttendanceDialog(QDialog):
    """Date picker + table view + CSV download for daily attendance."""

    _COLUMNS = ["Employee ID", "Name", "Department",
                "First Seen", "Last Seen", "Work Hours", "Status", "Camera", "Confidence"]

    def __init__(self, db: EmployeeDatabase, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self.setWindowTitle("Attendance Sheet")
        self.setModal(True)
        self.setMinimumSize(QSize(820, 520))
        self._auto_refresh = QTimer(self)
        self._auto_refresh.timeout.connect(self._load)
        self._auto_refresh.start(5_000)   # refresh every 5 seconds
        self.setStyleSheet(f"""
            QDialog   {{ background: {C_PANEL}; color: {C_TEXT};
                         font-family: 'Segoe UI', sans-serif; }}
            QLabel    {{ color: {C_TEXT}; font-size: 13px; }}
            QTableWidget {{
                background: {C_CARD}; color: {C_TEXT};
                gridline-color: {C_BORDER}; border: none;
                font-size: 12px;
            }}
            QTableWidget::item {{ padding: 6px 10px; }}
            QTableWidget::item:selected {{
                background: {C_ACCENT}; color: #fff;
            }}
            QHeaderView::section {{
                background: {C_BG}; color: {C_MUTED};
                border: none; border-bottom: 1px solid {C_BORDER};
                padding: 6px 10px; font-size: 11px; font-weight: 600;
            }}
            QDateEdit {{
                background: {C_CARD}; color: {C_TEXT};
                border: 1px solid {C_BORDER}; border-radius: 6px;
                padding: 6px 10px; font-size: 13px;
            }}
            QDateEdit::drop-down {{ border: none; width: 20px; }}
        """)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # Title
        ttl = QLabel("Attendance Sheet")
        ttl.setStyleSheet(
            f"font-size: 17px; font-weight: 700; color: {C_TEXT};")
        root.addWidget(ttl)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        date_lbl = QLabel("Date:")
        date_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 12px;")
        self._date_edit = QDateEdit(QDate.currentDate())
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setFixedWidth(140)
        self._date_edit.dateChanged.connect(self._load)

        refresh_btn = styled_btn("Refresh", C_CARD, "#475569")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self._load)

        self._count_lbl = QLabel("0 records")
        self._count_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 12px;")
        live_lbl = QLabel("● Live")
        live_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 11px; font-weight: 600;")

        download_btn = styled_btn("Download CSV", C_GREEN, "#16a34a")
        download_btn.clicked.connect(self._download)

        ctrl.addWidget(date_lbl)
        ctrl.addWidget(self._date_edit)
        ctrl.addWidget(refresh_btn)
        ctrl.addWidget(self._count_lbl)
        ctrl.addWidget(live_lbl)
        ctrl.addStretch()
        ctrl.addWidget(download_btn)
        root.addLayout(ctrl)

        # Table
        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            self._table.styleSheet() +
            f"alternate-background-color: {C_BG};"
        )
        root.addWidget(self._table, stretch=1)

        # Close
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = styled_btn("Close", C_CARD, "#475569")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _selected_date(self):
        qd = self._date_edit.date()
        from datetime import date as _date
        return _date(qd.year(), qd.month(), qd.day())

    def _load(self) -> None:
        records = self._db.get_attendance_by_date(self._selected_date())
        self._records = records
        self._table.setRowCount(0)
        for r in records:
            row = self._table.rowCount()
            self._table.insertRow(row)
            first = (r.get("first_seen") or "")[11:19]   # HH:MM:SS only
            last  = (r.get("last_seen")  or "")[11:19]
            conf  = f"{float(r.get('confidence', 0)) * 100:.1f}%"

            work_min = r.get("work_duration_minutes")
            if work_min is not None:
                h, m = divmod(work_min, 60)
                work_str = f"{h}h {m:02d}m"
            else:
                work_str = ""

            status = "Late" if r.get("is_late") else "On Time"

            for col, val in enumerate([
                r.get("employee_id", ""),
                r.get("employee_name", ""),
                r.get("department", ""),
                first, last,
                work_str,
                status,
                r.get("camera_id", ""),
                conf,
            ]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                if col == 6:  # Status column — colour-code
                    item.setForeground(
                        __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(
                            "#ef4444" if val == "Late" else "#22c55e"
                        )
                    )
                self._table.setItem(row, col, item)

        n = len(records)
        self._count_lbl.setText(f"{n} record{'s' if n != 1 else ''}")

    def closeEvent(self, event) -> None:
        self._auto_refresh.stop()
        super().closeEvent(event)

    def _download(self) -> None:
        from modules.attendance_exporter import build_csv_bytes
        target = self._selected_date()
        default_name = f"attendance_{target}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Attendance CSV",
            str(Path.home() / "Desktop" / default_name),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            records = self._db.get_attendance_by_date(target)
            Path(path).write_bytes(build_csv_bytes(records))
            QMessageBox.information(
                self, "Saved",
                f"Attendance saved to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not save file:\n{exc}")


# ─────────────────────────────────────────────────────────────
# Add Camera Dialog
# ─────────────────────────────────────────────────────────────

class AddCameraDialog(QDialog):
    def __init__(self, cfg: Dict, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self.setWindowTitle("Add Camera")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(f"""
            QDialog   {{ background: {C_PANEL}; color: {C_TEXT};
                          font-family: 'Segoe UI', sans-serif; }}
            QLabel    {{ color: {C_TEXT}; font-size: 13px; }}
            QLineEdit {{ background: {C_CARD}; color: {C_TEXT};
                          border: 1px solid {C_BORDER}; border-radius: 6px;
                          padding: 8px 12px; font-size: 13px; }}
            QLineEdit:focus {{ border-color: {C_ACCENT}; }}
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(22, 22, 22, 22)

        ttl = QLabel("Add New Camera")
        ttl.setStyleSheet(
            f"font-size: 17px; font-weight: 700; color: {C_TEXT};")
        root.addWidget(ttl)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._id_edit  = QLineEdit()
        self._id_edit.setPlaceholderText("cam_ip_02")
        self._nm_edit  = QLineEdit()
        self._nm_edit.setPlaceholderText("Back Entrance")
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText(
            "rtsp://admin:pass@192.168.1.x:554/Streaming/Channels/101")
        self._fps_edit = QLineEdit("25")

        form.addRow("Camera ID :", self._id_edit)
        form.addRow("Name :",      self._nm_edit)
        form.addRow("RTSP URL :",  self._url_edit)
        form.addRow("FPS :",       self._fps_edit)
        root.addLayout(form)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {C_AMBER}; font-size: 12px;")
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = styled_btn("Cancel", C_CARD, "#475569")
        cancel.clicked.connect(self.reject)
        ok = styled_btn("Add Camera", C_ACCENT, C_ACCENT_H)
        ok.clicked.connect(self._save)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        root.addLayout(btn_row)

    def _save(self) -> None:
        cam_id = "".join(
            c for c in self._id_edit.text().strip() if c.isalnum() or c in "-_"
        )
        name   = self._nm_edit.text().strip() or cam_id
        source = self._url_edit.text().strip()
        try:
            fps = max(1, min(60, int(self._fps_edit.text().strip() or "25")))
        except ValueError:
            fps = 25

        if not cam_id:
            self._status.setText("Camera ID is required.")
            return
        if not source:
            self._status.setText("RTSP URL is required.")
            return

        existing = [c.get("id") for c in self._cfg.get("cameras", [])]
        if cam_id in existing:
            self._status.setText(f"Camera '{cam_id}' already exists in config.")
            return

        try:
            cfg_path = BASE_DIR / "config" / "config.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            raw.setdefault("cameras", []).append(
                {"id": cam_id, "name": name, "source": source, "fps": fps}
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)
            self._cfg.setdefault("cameras", []).append(
                {"id": cam_id, "name": name, "source": source, "fps": fps}
            )
        except Exception as exc:
            self._status.setText(f"Save failed: {exc}")
            return

        self.accept()


# ─────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Employee Face Recognition")
        self.setMinimumSize(QSize(1280, 720))

        # ── Database ─────────────────────────────────────────
        db_cfg = cfg.get("database", {})
        if db_cfg.get("type", "sqlite") == "sqlite":
            db_url = ("sqlite:///" +
                      db_cfg.get("sqlite", {}).get("path", "data/employees.db"))
        else:
            pg = db_cfg.get("postgresql", {})
            db_url = (f"postgresql://{pg['user']}:{pg['password']}"
                      f"@{pg['host']}:{pg['port']}/{pg['name']}")

        self.db = EmployeeDatabase(db_url)
        self.db.initialize()

        # ── Camera worker ────────────────────────────────────
        self.worker = CameraWorker(cfg, self.db)
        self.worker.detection.connect(self._on_detection)
        self.worker.status.connect(self._on_status)
        self.worker.fps_update.connect(self.stats_bar.update_fps
                                       if hasattr(self, "stats_bar") else lambda _: None)

        self._build_ui()
        # Re-connect fps now that stats_bar exists
        self.worker.fps_update.connect(self.stats_bar.update_fps)

        # Load initial employee list + stats
        self._refresh_employees()
        self._refresh_stats()

        # Periodic stats refresh
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(5_000)

        # 30fps display timer — decoupled from recognition speed
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(33)
        self._display_timer.timeout.connect(self._refresh_display)
        self._display_timer.start()

        # Start recognition
        self.worker.start()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Title row
        trow = QHBoxLayout()
        ico = QLabel("◉")
        ico.setStyleSheet(f"color: {C_ACCENT}; font-size: 20px;")
        ttl = QLabel("Employee Face Recognition")
        ttl.setStyleSheet(
            f"color: {C_TEXT}; font-size: 18px; font-weight: 700;")
        self._status_lbl = QLabel("Starting…")
        self._status_lbl.setStyleSheet(
            f"color: {C_MUTED}; font-size: 12px;")
        att_btn = styled_btn("Attendance", C_GREEN, "#16a34a")
        att_btn.setFixedWidth(110)
        att_btn.clicked.connect(self._open_attendance)
        cam_btn = styled_btn("+ Camera", C_CARD, "#475569")
        cam_btn.setFixedWidth(110)
        cam_btn.clicked.connect(self._open_add_camera)

        trow.addWidget(ico)
        trow.addWidget(ttl)
        trow.addStretch()
        trow.addWidget(cam_btn)
        trow.addWidget(att_btn)
        trow.addWidget(self._status_lbl)
        root.addLayout(trow)

        # Stats bar
        self.stats_bar = StatsBar()
        root.addWidget(self.stats_bar)

        # Three-panel splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        self.emp_panel = EmployeePanel()
        self.emp_panel.add_requested.connect(self._open_add_dialog)
        self.emp_panel.delete_requested.connect(self._delete_employee)
        self.emp_panel.detail_requested.connect(self._show_detail)

        self.video = VideoDisplay()
        self.log_panel = LogPanel()

        splitter.addWidget(self.emp_panel)
        splitter.addWidget(self.video)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

    # ── Signal handlers ──────────────────────────────────────

    def _refresh_display(self) -> None:
        frame = self.worker.get_latest_annotated_frame()
        if frame is not None:
            self.video.update_frame(frame)

    def _on_detection(self, event: Dict) -> None:
        self.log_panel.add_event(event)

    def _on_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    # ── Data helpers ─────────────────────────────────────────

    def _refresh_employees(self) -> None:
        emps = self.db.get_all_employees()
        self.emp_panel.refresh(emps)

    def _refresh_stats(self) -> None:
        self.stats_bar.update_stats(self.db.get_detection_stats())

    # ── Actions ──────────────────────────────────────────────

    def _open_attendance(self) -> None:
        dlg = AttendanceDialog(self.db, self)
        dlg.exec()

    def _open_add_camera(self) -> None:
        dlg = AddCameraDialog(self.cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            QMessageBox.information(
                self, "Camera Added",
                "Camera saved to config.\nRestart the app to load the new camera.",
            )

    def _open_add_dialog(self) -> None:
        dlg = AddEmployeeDialog(self.worker, self.db, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_employees()
            self._refresh_stats()

    def _delete_employee(self, emp_id: str) -> None:
        emp = self.db.get_employee(emp_id)
        name = emp["name"] if emp else emp_id
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}'?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_employee(emp_id)
            self.worker.rebuild_index()
            self._refresh_employees()
            self._refresh_stats()

    def _show_detail(self, emp_id: str) -> None:
        emp = self.db.get_employee(emp_id)
        if emp is None:
            return
        dlg = EmployeeDetailDialog(emp, self)
        dlg.delete_requested.connect(self._on_detail_delete)
        dlg.exec()

    def _on_detail_delete(self, emp_id: str) -> None:
        self.db.delete_employee(emp_id)
        self.worker.rebuild_index()
        self._refresh_employees()
        self._refresh_stats()

    # ── Window close ─────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.worker.wait(4_000)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure all relative paths (config/models/data) resolve from project root.
    os.chdir(BASE_DIR)

    parser = argparse.ArgumentParser(
        description="Employee Face Recognition — Desktop App")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--camera", type=int, default=None,
                        help="Override camera source (webcam index, e.g. 0)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.camera is not None:
        if cfg.get("cameras"):
            cfg["cameras"][0]["source"] = args.camera
        else:
            cfg["cameras"] = [{"id": "cam_local", "source": args.camera}]

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    win = MainWindow(cfg)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
