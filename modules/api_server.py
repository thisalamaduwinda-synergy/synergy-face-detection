"""
api_server.py
─────────────────────────────────────────────────────────────
FastAPI application exposing:

  REST endpoints
  ──────────────
  POST   /api/employees            register new employee
  GET    /api/employees            list all employees
  GET    /api/employees/{id}       get one employee
  DELETE /api/employees/{id}       deactivate employee
  GET    /api/logs                 recent detection logs
  GET    /api/stats                system statistics
  GET    /api/cameras              camera status
  GET    /api/attendance           today's attendance
  GET    /api/attendance/{date}    attendance for a specific date (YYYY-MM-DD)

  Streaming
  ─────────
  GET    /video/{camera_id}        MJPEG live stream
  WS     /ws/events                real-time detection events (JSON push)
  WS     /ws/stream/{camera_id}    annotated JPEG frames via WebSocket

  Dashboard
  ─────────
  GET    /                         serve the HTML dashboard
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket,
    WebSocketDisconnect, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

logger = logging.getLogger(__name__)

_SERVER_START_TIME = time.time()

# ─────────────────────────────────────────────────────────────
# Auth helpers  (stdlib only — no extra dependencies)
# ─────────────────────────────────────────────────────────────

_TOKEN_TTL = 86_400          # 24 hours
_bearer = HTTPBearer(auto_error=False)


def _make_secret(password: str) -> bytes:
    """Derive a stable HMAC signing key from the admin password."""
    return hashlib.sha256(f"frs-auth:{password}".encode()).digest()


def _issue_token(password: str) -> str:
    """Create a <expiry>.<hmac> token."""
    expiry = int(time.time()) + _TOKEN_TTL
    key = _make_secret(password)
    sig = hmac.new(key, str(expiry).encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def _verify_token(token: str, password: str) -> bool:
    """Return True if token is valid and not expired."""
    try:
        expiry_str, sig = token.split(".", 1)
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        return False
    if time.time() > expiry:
        return False
    key = _make_secret(password)
    expected = hmac.new(key, expiry_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)

# ─────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────

def create_app(
    db,               # EmployeeDatabase
    recognizer,       # FaceRecognizer
    camera_manager,   # MultiCameraManager
    logging_svc,      # LoggingService
    face_detector,    # FaceDetector (for annotation in stream)
    config: Dict,
    attendance_exporter=None,  # AttendanceExporter (optional)
) -> FastAPI:

    app = FastAPI(
        title="Employee Face Recognition API",
        version="1.0.0",
        description="Real-time employee identification via CCTV streams.",
    )

    # ── Auth configuration ───────────────────────────────────
    _admin_password: str = os.environ.get("ADMIN_PASSWORD", "")
    _auth_enabled: bool = bool(_admin_password)

    def require_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
        _token: Optional[str] = Query(default=None),
    ) -> None:
        if not _auth_enabled:
            return
        # Accept token from Authorization header OR ?_token= query param (for file downloads)
        token = (credentials.credentials if credentials else None) or _token or ""
        if not token or not _verify_token(token, _admin_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── CORS ────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files & templates ────────────────────────────
    BASE_DIR = Path(__file__).resolve().parent.parent
    templates_dir = BASE_DIR / "dashboard" / "templates"
    static_dir    = BASE_DIR / "dashboard" / "static"

    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))
    else:
        templates = None

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── WebSocket connection registry ────────────────────────
    _event_clients: List[WebSocket] = []
    _event_lock = threading.Lock()
    _face_det_lock = threading.Lock()
    _loop: Optional[asyncio.AbstractEventLoop] = None

    # Dedicated detector for registration to avoid concurrent state races
    # with the live camera processing pipeline.
    _reg_face_detector = face_detector
    try:
        from modules.face_detector import FaceDetector
        _reg_face_detector = FaceDetector(
            yunet_model=getattr(face_detector, "yunet_model", "data/models/face_detection_yunet_2023mar.onnx"),
            sface_model=getattr(face_detector, "sface_model", "data/models/face_recognition_sface_2021dec.onnx"),
            det_thresh=float(getattr(face_detector, "det_thresh", 0.6)),
            nms_thresh=float(getattr(face_detector, "nms_thresh", 0.3)),
            min_face_size=int(getattr(face_detector, "min_face_size", 40)),
            backend=getattr(face_detector, "backend", "insightface"),
            insightface_model=getattr(face_detector, "insightface_model", "buffalo_l"),
            insightface_root=getattr(face_detector, "insightface_root", "data/models/insightface"),
            insightface_det_size=cast(Tuple[int, int], tuple(getattr(face_detector, "insightface_det_size", (640, 640)))),
        )
        _reg_face_detector.initialize()
    except Exception as exc:
        logger.warning("Falling back to shared detector for registration: %s", exc)

    def _detect_faces_for_registration(
        image: np.ndarray,
        det_thresh: float,
        min_face_size: int,
    ) -> List:
        """Run detection with temporary settings, then restore detector state."""
        with _face_det_lock:
            old_thresh = float(_reg_face_detector.det_thresh)
            old_min_face = int(_reg_face_detector.min_face_size)
            old_model_thresh = None

            try:
                _inner_detector = getattr(_reg_face_detector, "_detector", None)
                if _inner_detector is not None:
                    old_model_thresh = float(_inner_detector.getScoreThreshold())
                    _inner_detector.setScoreThreshold(float(det_thresh))

                _reg_face_detector.det_thresh = float(det_thresh)
                _reg_face_detector.min_face_size = int(min_face_size)
                return _reg_face_detector.detect(image)
            finally:
                _reg_face_detector.det_thresh = old_thresh
                _reg_face_detector.min_face_size = old_min_face
                _restore_detector = getattr(_reg_face_detector, "_detector", None)
                if old_model_thresh is not None and _restore_detector is not None:
                    _restore_detector.setScoreThreshold(old_model_thresh)

    def _extract_embedding_best_effort(image: np.ndarray) -> Optional[np.ndarray]:
        """Try multiple preprocessing strategies for robust photo registration."""
        attempts = [(image, 0.60, 40)]

        h, w = image.shape[:2]
        if max(h, w) < 1200:
            resized = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            attempts.append((resized, 0.45, 20))
        else:
            attempts.append((image, 0.45, 20))

        img_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        img_ycc[:, :, 0] = cv2.equalizeHist(img_ycc[:, :, 0])
        enhanced = cv2.cvtColor(img_ycc, cv2.COLOR_YCrCb2BGR)
        attempts.append((enhanced, 0.45, 20))

        for frame, det_thresh, min_face_size in attempts:
            faces = _detect_faces_for_registration(frame, det_thresh, min_face_size)
            if faces and faces[0].embedding is not None:
                return faces[0].embedding
        return None

    def _average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
        """Average multiple embeddings and return a single L2-normalized vector."""
        if not embeddings:
            raise ValueError("No embeddings to average.")
        stacked = np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings])
        avg = stacked.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        return avg

    def _broadcast_event(event: Dict) -> None:
        """Called by LoggingService when a new detection is persisted."""
        if _loop is None or not _loop.is_running():
            return
        with _event_lock:
            dead = []
            for ws in list(_event_clients):
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps(event)), _loop
                    )
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _event_clients.remove(ws)

    logging_svc.subscribe(_broadcast_event)

    @app.on_event("startup")
    async def _capture_loop():
        nonlocal _loop
        _loop = asyncio.get_event_loop()

    # ═══════════════════════════════════════════════════════
    # Dashboard route
    # ═══════════════════════════════════════════════════════

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        if templates:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "config": config.get("system", {})},
            )
        return HTMLResponse("<h1>Dashboard templates not found.</h1>", status_code=200)

    # ═══════════════════════════════════════════════════════
    # Auth endpoints  (no token required)
    # ═══════════════════════════════════════════════════════

    @app.get("/api/auth/config")
    async def auth_config():
        """Tell the dashboard whether authentication is required."""
        return {"require_auth": _auth_enabled}

    @app.post("/api/auth/login")
    async def auth_login(request: Request):
        body = await request.json()
        password = body.get("password", "")
        if not _auth_enabled:
            return {"token": "", "expires_in": _TOKEN_TTL, "message": "Auth disabled"}
        if password != _admin_password:
            raise HTTPException(status_code=401, detail="Incorrect password.")
        token = _issue_token(_admin_password)
        return {"token": token, "expires_in": _TOKEN_TTL}

    @app.get("/api/auth/check")
    async def auth_check(_: None = Depends(require_auth)):
        return {"authenticated": True}

    # ═══════════════════════════════════════════════════════
    # Health endpoint  (no auth — safe for monitoring tools)
    # ═══════════════════════════════════════════════════════

    @app.get("/api/health")
    async def health():
        """System health check. Returns 200 when operational."""
        import datetime as _dt
        uptime_s = int(time.time() - _SERVER_START_TIME)
        h, rem  = divmod(uptime_s, 3600)
        m, s    = divmod(rem, 60)

        cam_stats  = camera_manager.get_all_stats()
        cam_ok     = sum(1 for c in cam_stats.values() if c.get("connected"))
        cam_total  = len(cam_stats)

        db_ok = False
        emp_count = 0
        try:
            emp_count = db.employee_count()
            db_ok = True
        except Exception:
            pass

        today_att = 0
        try:
            today_att = len(db.get_attendance_by_date())
        except Exception:
            pass

        return {
            "status": "ok",
            "uptime": f"{h:02d}:{m:02d}:{s:02d}",
            "uptime_seconds": uptime_s,
            "server_time": _dt.datetime.now().isoformat(),
            "database": {"ok": db_ok, "employees": emp_count},
            "cameras": {
                "total": cam_total,
                "connected": cam_ok,
                "disconnected": cam_total - cam_ok,
            },
            "recognition": {
                "indexed_employees": recognizer.employee_count,
                "threshold": recognizer.threshold,
            },
            "attendance_today": today_att,
        }

    # ═══════════════════════════════════════════════════════
    # Employee endpoints
    # ═══════════════════════════════════════════════════════

    @app.post("/api/employees", status_code=status.HTTP_201_CREATED)
    async def register_employee(
        employee_id: str = Form(...),
        name: str = Form(...),
        department: str = Form(""),
        photos: Optional[List[UploadFile]] = File(None),
        photo: Optional[UploadFile] = File(None),
        _: None = Depends(require_auth),
    ):
        """Register a new employee using 1 to 5 face photos."""
        photo_path = ""
        embeddings: List[np.ndarray] = []

        uploaded_photos: List[UploadFile] = list(photos or [])
        if photo is not None:
            uploaded_photos.append(photo)

        if len(uploaded_photos) < 1:
            raise HTTPException(
                status_code=422,
                detail="At least 1 photo is required.",
            )
        if len(uploaded_photos) > 5:
            raise HTTPException(
                status_code=422,
                detail="A maximum of 5 photos is allowed.",
            )

        photo_dir = Path("data/employees")
        photo_dir.mkdir(parents=True, exist_ok=True)
        allowed = {"image/jpeg", "image/png", "image/jpg"}
        safe_id = "".join(c for c in employee_id if c.isalnum() or c in "-_")

        for idx, uploaded in enumerate(uploaded_photos, start=1):
            if uploaded.content_type not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG/PNG images are accepted.",
                )

            img_bytes = await uploaded.read()
            img_buf = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not decode image #{idx}.",
                )

            emb = _extract_embedding_best_effort(img)
            if emb is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"No face detected in image #{idx}. "
                        "Use clear front-facing photos with good lighting."
                    ),
                )
            embeddings.append(emb)

            saved_photo_path = str(photo_dir / f"{safe_id}_{idx}.jpg")
            cv2.imwrite(saved_photo_path, img)
            if idx == 1:
                photo_path = saved_photo_path

        embedding = _average_embeddings(embeddings)

        try:
            emp = db.add_employee(
                employee_id=employee_id,
                name=name,
                department=department,
                embedding=embedding,
                photo_path=photo_path,
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        # Rebuild FAISS index
        all_employees = db.get_all_employees_with_embeddings()
        recognizer.build_index(all_employees)

        return {"message": "Employee registered successfully.", "employee_id": employee_id}

    @app.get("/api/employees")
    async def list_employees(active_only: bool = True, _: None = Depends(require_auth)):
        employees = db.get_all_employees()
        return {"employees": employees, "count": len(employees)}

    @app.get("/api/employees/{employee_id}")
    async def get_employee(employee_id: str, _: None = Depends(require_auth)):
        emp = db.get_employee(employee_id)
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found.")
        return emp

    @app.get("/api/employees/{employee_id}/photo")
    async def get_employee_photo(employee_id: str, _: None = Depends(require_auth)):
        """Return the employee's face photo (JPEG/PNG). 404 if no photo stored."""
        emp = db.get_employee(employee_id)
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found.")
        photo_path = emp.get("photo_path", "")
        if not photo_path or not Path(photo_path).is_file():
            raise HTTPException(status_code=404, detail="No photo available.")
        suffix = Path(photo_path).suffix.lower()
        media_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
        return StreamingResponse(
            open(photo_path, "rb"),
            media_type=media_type,
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.put("/api/employees/{employee_id}", status_code=status.HTTP_200_OK)
    async def update_employee(
        employee_id: str,
        name: Optional[str] = Form(None),
        department: Optional[str] = Form(None),
        photos: Optional[List[UploadFile]] = File(None),
        _: None = Depends(require_auth),
    ):
        """Update an employee's name, department, and optionally replace face photos."""
        if name is None and department is None and not photos:
            raise HTTPException(status_code=400, detail="Provide at least one field to update.")

        updated_embedding: Optional[Any] = None
        if photos:
            embeddings: List[Any] = []
            for idx, uploaded in enumerate(photos[:5], start=1):
                if uploaded.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
                    raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")
                img_bytes = await uploaded.read()
                img_buf = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise HTTPException(status_code=400, detail=f"Could not decode image #{idx}.")
                emb = _extract_embedding_best_effort(img)
                if emb is None:
                    raise HTTPException(status_code=422, detail=f"No face detected in image #{idx}.")
                embeddings.append(emb)
            updated_embedding = _average_embeddings(embeddings)

        success = db.update_employee(employee_id, name=name, department=department)
        if not success:
            raise HTTPException(status_code=404, detail="Employee not found.")

        if updated_embedding is not None:
            db.update_employee_embedding(employee_id, updated_embedding)

        all_employees = db.get_all_employees_with_embeddings()
        recognizer.build_index(all_employees)
        return {"message": "Employee updated successfully.", "employee_id": employee_id}

    @app.delete("/api/employees/{employee_id}", status_code=status.HTTP_200_OK)
    async def deactivate_employee(employee_id: str, _: None = Depends(require_auth)):
        success = db.deactivate_employee(employee_id)
        if not success:
            raise HTTPException(status_code=404, detail="Employee not found.")
        # Rebuild index without this employee
        all_employees = db.get_all_employees_with_embeddings()
        recognizer.build_index(all_employees)
        return {"message": f"Employee {employee_id} deactivated."}

    # ═══════════════════════════════════════════════════════
    # Detection log endpoints
    # ═══════════════════════════════════════════════════════

    @app.get("/api/logs")
    async def get_logs(
        limit: int = 100,
        camera_id: Optional[str] = None,
        unknown_only: bool = False,
        _: None = Depends(require_auth),
    ):
        if limit > 1000:
            limit = 1000  # Hard cap
        logs = db.get_recent_logs(
            limit=limit, camera_id=camera_id, only_unknown=unknown_only
        )
        return {"logs": logs, "count": len(logs)}

    @app.get("/api/stats")
    async def get_stats(_: None = Depends(require_auth)):
        db_stats = db.get_detection_stats()
        cam_stats = camera_manager.get_all_stats()
        recognizer_info = {
            "indexed_employees": recognizer.employee_count,
            "threshold": recognizer.threshold,
        }
        return {
            "database": db_stats,
            "cameras": cam_stats,
            "recognizer": recognizer_info,
        }

    @app.get("/api/cameras")
    async def get_cameras(_: None = Depends(require_auth)):
        return camera_manager.get_all_stats()

    @app.post("/api/cameras", status_code=status.HTTP_201_CREATED)
    async def add_camera_api(request: Request, _: None = Depends(require_auth)):
        """Add a new camera stream at runtime and persist to config.yaml."""
        body = await request.json()
        cam_id = "".join(c for c in str(body.get("id", "")).strip() if c.isalnum() or c in "-_")
        source = str(body.get("source", "")).strip()
        name   = str(body.get("name", cam_id)).strip() or cam_id
        fps    = max(1, min(60, int(body.get("fps", 25))))

        if not cam_id:
            raise HTTPException(status_code=400, detail="id is required.")
        if not source:
            raise HTTPException(status_code=400, detail="source (RTSP URL) is required.")
        if camera_manager.get_camera(cam_id) is not None:
            raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' already exists.")

        stream = camera_manager.add_camera(cam_id, source, fps=fps)
        stream.start()

        try:
            import yaml as _yaml
            cfg_path = BASE_DIR / "config" / "config.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = _yaml.safe_load(f)
            raw_cfg.setdefault("cameras", []).append(
                {"id": cam_id, "name": name, "source": source, "fps": fps}
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                _yaml.dump(raw_cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            logger.warning("Camera added to runtime but config save failed: %s", exc)

        return {"message": f"Camera '{cam_id}' added.", "camera_id": cam_id, "name": name}

    @app.delete("/api/cameras/{camera_id}", status_code=status.HTTP_200_OK)
    async def remove_camera_api(camera_id: str, _: None = Depends(require_auth)):
        """Remove a camera stream at runtime and remove from config.yaml."""
        if not camera_manager.remove_camera(camera_id):
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

        try:
            import yaml as _yaml
            cfg_path = BASE_DIR / "config" / "config.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = _yaml.safe_load(f)
            raw_cfg["cameras"] = [
                c for c in raw_cfg.get("cameras", []) if c.get("id") != camera_id
            ]
            with open(cfg_path, "w", encoding="utf-8") as f:
                _yaml.dump(raw_cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            logger.warning("Camera removed from runtime but config save failed: %s", exc)

        return {"message": f"Camera '{camera_id}' removed."}

    # ═══════════════════════════════════════════════════════
    # Attendance endpoints
    # ═══════════════════════════════════════════════════════

    @app.get("/api/attendance")
    async def get_attendance_today(_: None = Depends(require_auth)):
        """Return today's attendance list."""
        records = db.get_attendance_by_date()
        return {"date": str(__import__("datetime").date.today()), "records": records, "count": len(records)}

    @app.get("/api/attendance/{target_date}")
    async def get_attendance_by_date(target_date: str, _: None = Depends(require_auth)):
        """Return attendance for a specific date (YYYY-MM-DD)."""
        try:
            from datetime import date as _date
            d = _date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD format.")
        records = db.get_attendance_by_date(d)
        return {"date": target_date, "records": records, "count": len(records)}

    @app.get("/api/attendance/export/today")
    async def export_attendance_today(_: None = Depends(require_auth)):
        """Download today's attendance as a CSV file."""
        from modules.attendance_exporter import build_csv_bytes
        records = db.get_attendance_by_date()
        csv_bytes = build_csv_bytes(records)
        from datetime import date as _date
        filename = f"attendance_{_date.today()}.csv"
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/export/{target_date}")
    async def export_attendance_by_date(target_date: str, _: None = Depends(require_auth)):
        """Download attendance for a specific date (YYYY-MM-DD) as a CSV file."""
        try:
            from datetime import date as _date
            d = _date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD format.")
        from modules.attendance_exporter import build_csv_bytes
        records = db.get_attendance_by_date(d)
        csv_bytes = build_csv_bytes(records)
        filename = f"attendance_{target_date}.csv"
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/absent")
    async def get_absent_today(_: None = Depends(require_auth)):
        """Return all active employees who have NOT been seen today."""
        records = db.get_absent_employees()
        return {
            "date": str(__import__("datetime").date.today()),
            "absent": records,
            "count": len(records),
        }

    @app.get("/api/attendance/absent/{target_date}")
    async def get_absent_by_date(target_date: str, _: None = Depends(require_auth)):
        """Return absent employees for a specific date (YYYY-MM-DD)."""
        try:
            from datetime import date as _date
            d = _date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD format.")
        records = db.get_absent_employees(d)
        return {"date": target_date, "absent": records, "count": len(records)}

    @app.get("/api/attendance/range")
    async def get_attendance_range(start: str, end: str, employee_id: Optional[str] = None, _: None = Depends(require_auth)):
        """Return attendance records between start and end dates (YYYY-MM-DD)."""
        try:
            from datetime import date as _date
            s = _date.fromisoformat(start)
            e = _date.fromisoformat(end)
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format.")
        if s > e:
            raise HTTPException(status_code=400, detail="start must be <= end.")
        records = db.get_attendance_range(s, e, employee_id=employee_id)
        return {"start": start, "end": end, "records": records, "count": len(records)}

    @app.get("/api/attendance/export/range")
    async def export_attendance_range(start: str, end: str, employee_id: Optional[str] = None, _: None = Depends(require_auth)):
        """Download attendance CSV for a date range (YYYY-MM-DD)."""
        try:
            from datetime import date as _date
            s = _date.fromisoformat(start)
            e = _date.fromisoformat(end)
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format.")
        if s > e:
            raise HTTPException(status_code=400, detail="start must be <= end.")
        from modules.attendance_exporter import build_csv_bytes
        records = db.get_attendance_range(s, e, employee_id=employee_id)
        csv_bytes = build_csv_bytes(records)
        filename = f"attendance_{start}_to_{end}.csv"
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/export/today/excel")
    async def export_attendance_today_excel(_: None = Depends(require_auth)):
        """Download today's attendance as a formatted Excel file."""
        from modules.attendance_exporter import build_excel_bytes
        import datetime as _dt
        records = db.get_attendance_by_date()
        xl_bytes = build_excel_bytes(records)
        filename = f"attendance_{_dt.date.today()}.xlsx"
        return StreamingResponse(
            iter([xl_bytes]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/export/{target_date}/excel")
    async def export_attendance_date_excel(target_date: str, _: None = Depends(require_auth)):
        """Download attendance for a specific date as a formatted Excel file."""
        try:
            from datetime import date as _date
            d = _date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD format.")
        from modules.attendance_exporter import build_excel_bytes
        records = db.get_attendance_by_date(d)
        xl_bytes = build_excel_bytes(records)
        filename = f"attendance_{target_date}.xlsx"
        return StreamingResponse(
            iter([xl_bytes]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/export/range/excel")
    async def export_attendance_range_excel(start: str, end: str, _: None = Depends(require_auth)):
        """Download attendance for a date range as a formatted Excel file."""
        try:
            from datetime import date as _date
            s, e = _date.fromisoformat(start), _date.fromisoformat(end)
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format.")
        if s > e:
            raise HTTPException(status_code=400, detail="start must be <= end.")
        from modules.attendance_exporter import build_excel_bytes
        records = db.get_attendance_range(s, e)
        xl_bytes = build_excel_bytes(records)
        filename = f"attendance_{start}_to_{end}.xlsx"
        return StreamingResponse(
            iter([xl_bytes]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/monthly")
    async def get_monthly_summary(year: int, month: int, _: None = Depends(require_auth)):
        """Return per-employee attendance summary for the given year/month."""
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="month must be 1–12.")
        records = db.get_monthly_summary(year, month)
        return {"year": year, "month": month, "summary": records, "count": len(records)}

    @app.get("/api/attendance/export/monthly/excel")
    async def export_monthly_excel(year: int, month: int, _: None = Depends(require_auth)):
        """Download monthly attendance summary as a formatted Excel file."""
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="month must be 1–12.")
        from modules.attendance_exporter import build_monthly_excel_bytes
        records = db.get_monthly_summary(year, month)
        xl_bytes = build_monthly_excel_bytes(records)
        filename = f"attendance_monthly_{year}_{month:02d}.xlsx"
        return StreamingResponse(
            iter([xl_bytes]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/api/attendance/export/monthly")
    async def export_monthly_summary(year: int, month: int, _: None = Depends(require_auth)):
        """Download monthly attendance summary as a CSV file."""
        if not (1 <= month <= 12):
            raise HTTPException(status_code=400, detail="month must be 1–12.")
        from modules.attendance_exporter import build_monthly_csv_bytes
        records = db.get_monthly_summary(year, month)
        csv_bytes = build_monthly_csv_bytes(records)
        filename = f"attendance_monthly_{year}_{month:02d}.csv"
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ═══════════════════════════════════════════════════════
    # MJPEG video stream
    # ═══════════════════════════════════════════════════════

    @app.get("/video/{camera_id}")
    async def mjpeg_stream(camera_id: str):
        cam = camera_manager.get_camera(camera_id)
        if cam is None:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

        async def frame_generator():
            boundary = b"--frame\r\n"
            header = b"Content-Type: image/jpeg\r\n\r\n"
            while True:
                frame_obj = cam.read(timeout=0.1)
                if frame_obj is None:
                    await asyncio.sleep(0.05)
                    continue

                ret, jpeg = cv2.imencode(
                    ".jpg", frame_obj.frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75],
                )
                if not ret:
                    continue

                yield boundary + header + jpeg.tobytes() + b"\r\n"
                await asyncio.sleep(1 / 25)   # cap at 25 fps for streaming

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ═══════════════════════════════════════════════════════
    # WebSocket – real-time detection events
    # ═══════════════════════════════════════════════════════

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        nonlocal _loop
        _loop = asyncio.get_event_loop()
        await websocket.accept()
        with _event_lock:
            _event_clients.append(websocket)
        # Send recent events on connect
        recent = logging_svc.get_recent_events(limit=20)
        for ev in reversed(recent):
            try:
                await websocket.send_text(json.dumps(ev))
            except Exception:
                break
        try:
            while True:
                # Keep alive – client sends ping
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            with _event_lock:
                try:
                    _event_clients.remove(websocket)
                except ValueError:
                    pass

    # ═══════════════════════════════════════════════════════
    # WebSocket – annotated frame stream per camera
    # ═══════════════════════════════════════════════════════

    @app.websocket("/ws/stream/{camera_id}")
    async def ws_frame_stream(websocket: WebSocket, camera_id: str):
        cam = camera_manager.get_camera(camera_id)
        if cam is None:
            await websocket.close(code=1008, reason="Camera not found")
            return

        await websocket.accept()
        try:
            while True:
                frame_obj = cam.read(timeout=0.1)
                if frame_obj is None:
                    await asyncio.sleep(0.05)
                    continue

                ret, jpeg = cv2.imencode(
                    ".jpg", frame_obj.frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                )
                if ret:
                    await websocket.send_bytes(jpeg.tobytes())
                await asyncio.sleep(1 / 15)   # 15 fps over WS
        except WebSocketDisconnect:
            pass

    return app
