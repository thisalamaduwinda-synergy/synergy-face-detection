"""
employee_database.py
─────────────────────────────────────────────────────────────
SQLAlchemy-backed employee registry.

Supports:
  • SQLite  (default – zero-config, great for ≤1000 employees)
  • PostgreSQL (set DB_TYPE=postgresql in .env)

Schema:
  employees       – employee records + raw face embeddings
  detection_logs  – every recognition event
  attendance_logs – daily attendance (one row per employee per day)
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer,
    LargeBinary, String, Text, UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ORM models
# ─────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    department = Column(String(128), default="")
    company_id = Column(String(64), default="", index=True)
    photo_path = Column(String(512), default="")
    # 512-d float32 embedding stored as raw bytes (2048 B per employee)
    face_embedding = Column(LargeBinary, nullable=True)
    registered_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    is_vip = Column(Boolean, default=False)

    def to_dict(self, include_embedding: bool = False) -> Dict:
        data = {
            "id": self.id,
            "employee_id": self.employee_id,
            "name": self.name,
            "department": self.department,
            "company_id": self.company_id,
            "photo_path": self.photo_path,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "is_active": self.is_active,
            "is_vip": bool(self.is_vip),
            "has_embedding": self.face_embedding is not None,
        }
        if include_embedding and self.face_embedding is not None:
            emb = np.frombuffer(self.face_embedding, dtype=np.float32).copy()
            data["face_embedding"] = emb
        return data


class VIPVisitLog(Base):
    __tablename__ = "vip_visit_logs"
    __table_args__ = (
        UniqueConstraint("employee_id", "date", name="uq_vip_visit_per_day"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), nullable=False, index=True)
    employee_name = Column(String(128), nullable=False)
    department = Column(String(128), default="")
    company_id = Column(String(64), default="", index=True)
    date = Column(Date, nullable=False, index=True)
    in_time = Column(DateTime, nullable=False)
    out_time = Column(DateTime, nullable=False)
    camera_id = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)

    def to_dict(self) -> Dict:
        duration: Optional[int] = None
        if self.in_time and self.out_time:
            duration = int((self.out_time - self.in_time).total_seconds() / 60)
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "department": self.department,
            "company_id": self.company_id,
            "date": self.date.isoformat() if self.date else None,
            "in_time": self.in_time.isoformat() if self.in_time else None,
            "out_time": self.out_time.isoformat() if self.out_time else None,
            "camera_id": self.camera_id,
            "confidence": self.confidence,
            "visit_duration_minutes": duration,
        }


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    employee_id = Column(String(64), nullable=True, index=True)   # NULL → unknown
    employee_name = Column(String(128), default="Unknown")
    confidence = Column(Float, nullable=False)
    is_known = Column(Boolean, default=False)
    bbox_x1 = Column(Integer, default=0)
    bbox_y1 = Column(Integer, default=0)
    bbox_x2 = Column(Integer, default=0)
    bbox_y2 = Column(Integer, default=0)
    frame_path = Column(String(512), default="")  # optional saved frame

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "confidence": self.confidence,
            "is_known": self.is_known,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "frame_path": self.frame_path,
        }


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    __table_args__ = (
        UniqueConstraint("employee_id", "date", name="uq_attendance_per_day"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), nullable=False, index=True)
    employee_name = Column(String(128), nullable=False)
    department = Column(String(128), default="")
    date = Column(Date, nullable=False, index=True)
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    camera_id = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    is_late = Column(Boolean, default=False)

    def to_dict(self) -> Dict:
        work_minutes: Optional[int] = None
        if self.first_seen and self.last_seen:
            delta = self.last_seen - self.first_seen
            work_minutes = int(delta.total_seconds() / 60)
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "department": self.department,
            "date": self.date.isoformat() if self.date else None,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "camera_id": self.camera_id,
            "confidence": self.confidence,
            "is_late": bool(self.is_late),
            "work_duration_minutes": work_minutes,
        }


# ─────────────────────────────────────────────────────────────
# Database manager
# ─────────────────────────────────────────────────────────────

class EmployeeDatabase:
    """
    High-level interface to the employee registry.

    Usage
    -----
    db = EmployeeDatabase("sqlite:///data/employees.db")
    db.initialize()
    db.add_employee("EMP001", "Alice Smith", "Engineering", embedding)
    employees = db.get_all_employees_with_embeddings()
    """

    def __init__(self, database_url: str = "sqlite:///data/employees.db") -> None:
        self.database_url = database_url
        self._engine = None
        self._Session = None

    # ── Setup ───────────────────────────────────────────────

    def initialize(self) -> None:
        """Create engine, run migrations, and create tables if absent."""
        # Ensure parent directory exists (for SQLite)
        if self.database_url.startswith("sqlite"):
            db_path = self.database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        connect_args = {}
        if self.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._run_migrations()
        logger.info("Database initialised: %s", self.database_url)

    def _run_migrations(self) -> None:
        """Add columns introduced after initial release to existing databases."""
        migrations = [
            ("attendance_logs", "ALTER TABLE attendance_logs ADD COLUMN is_late BOOLEAN DEFAULT 0"),
            ("employees", "ALTER TABLE employees ADD COLUMN is_vip BOOLEAN DEFAULT 0"),
            ("employees", "ALTER TABLE employees ADD COLUMN company_id VARCHAR(64) DEFAULT ''"),
        ]
        with self._engine.connect() as conn:
            for table, sql in migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    logger.info("Migration applied: %s", sql[:60])
                except Exception:
                    pass  # column already exists

    def _session(self) -> Session:
        if self._Session is None:
            raise RuntimeError("Call initialize() first.")
        return self._Session()

    # ── Employee CRUD ───────────────────────────────────────

    def add_employee(
        self,
        employee_id: str,
        name: str,
        department: str = "",
        embedding: Optional[np.ndarray] = None,
        photo_path: str = "",
    ) -> Employee:
        """Insert a new employee. Raises ValueError if employee_id already exists."""
        embedding_bytes = None
        if embedding is not None:
            emb = embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            embedding_bytes = emb.tobytes()

        with self._session() as session:
            # Security: check for duplicates before insert
            existing = session.query(Employee).filter_by(employee_id=employee_id).first()
            if existing:
                raise ValueError(f"Employee '{employee_id}' already exists.")

            emp = Employee(
                employee_id=employee_id,
                name=name,
                department=department,
                face_embedding=embedding_bytes,
                photo_path=photo_path,
                registered_at=datetime.now(),
                is_active=True,
            )
            session.add(emp)
            session.commit()
            logger.info("Employee added: %s (%s)", name, employee_id)
            return emp

    def update_employee_embedding(
        self, employee_id: str, embedding: np.ndarray
    ) -> bool:
        """Replace the stored embedding for an existing employee."""
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        with self._session() as session:
            emp = session.query(Employee).filter_by(employee_id=employee_id).first()
            if emp is None:
                logger.warning("Employee not found: %s", employee_id)
                return False
            emp.face_embedding = emb.tobytes()
            session.commit()
            logger.info("Embedding updated for %s", employee_id)
            return True

    def deactivate_employee(self, employee_id: str) -> bool:
        """Soft-delete: set is_active=False."""
        with self._session() as session:
            emp = session.query(Employee).filter_by(employee_id=employee_id).first()
            if emp is None:
                return False
            emp.is_active = False
            session.commit()
            logger.info("Employee deactivated: %s", employee_id)
            return True

    def delete_employee(self, employee_id: str) -> bool:
        """Hard delete an employee record."""
        with self._session() as session:
            emp = session.query(Employee).filter_by(employee_id=employee_id).first()
            if emp is None:
                return False
            session.delete(emp)
            session.commit()
            logger.info("Employee deleted: %s", employee_id)
            return True

    def get_employee(self, employee_id: str) -> Optional[Dict]:
        with self._session() as session:
            emp = session.query(Employee).filter_by(
                employee_id=employee_id, is_active=True
            ).first()
            return emp.to_dict() if emp else None

    def get_all_employees(self) -> List[Dict]:
        """Return all active employees WITHOUT embeddings."""
        with self._session() as session:
            emps = session.query(Employee).filter_by(is_active=True).all()
            return [e.to_dict(include_embedding=False) for e in emps]

    def get_all_employees_with_embeddings(self) -> List[Dict]:
        """Return active employees WITH numpy embeddings for FAISS indexing."""
        with self._session() as session:
            emps = session.query(Employee).filter_by(is_active=True).all()
            result = []
            for e in emps:
                d = e.to_dict(include_embedding=True)
                result.append(d)
            return result

    def employee_count(self) -> int:
        with self._session() as session:
            return session.query(Employee).filter_by(is_active=True).count()

    # ── Detection log ───────────────────────────────────────

    def log_detection(
        self,
        camera_id: str,
        employee_id: Optional[str],
        employee_name: str,
        confidence: float,
        is_known: bool,
        bbox: Optional[List[int]] = None,
        frame_path: str = "",
        timestamp: Optional[datetime] = None,
    ) -> DetectionLog:
        """Insert a detection event into the log."""
        bbox = bbox or [0, 0, 0, 0]
        with self._session() as session:
            entry = DetectionLog(
                timestamp=timestamp or datetime.now(),
                camera_id=camera_id,
                employee_id=employee_id,
                employee_name=employee_name,
                confidence=round(float(confidence), 4),
                is_known=is_known,
                bbox_x1=bbox[0],
                bbox_y1=bbox[1],
                bbox_x2=bbox[2] if len(bbox) > 2 else 0,
                bbox_y2=bbox[3] if len(bbox) > 3 else 0,
                frame_path=frame_path,
            )
            session.add(entry)
            session.commit()
            return entry

    def get_recent_logs(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None,
        only_unknown: bool = False,
    ) -> List[Dict]:
        with self._session() as session:
            q = session.query(DetectionLog)
            if camera_id:
                q = q.filter(DetectionLog.camera_id == camera_id)
            if only_unknown:
                q = q.filter(DetectionLog.is_known == False)  # noqa
            logs = q.order_by(DetectionLog.timestamp.desc()).limit(limit).all()
            return [log.to_dict() for log in logs]

    def get_detection_stats(self) -> Dict:
        """Aggregate counts for the dashboard."""
        with self._session() as session:
            total = session.query(DetectionLog).count()
            known = session.query(DetectionLog).filter_by(is_known=True).count()
            unknown = total - known
            employees = session.query(Employee).filter_by(is_active=True).count()
        return {
            "total_detections": total,
            "known_detections": known,
            "unknown_detections": unknown,
            "registered_employees": employees,
        }

    # ── Attendance ──────────────────────────────────────────

    @staticmethod
    def _cap_to_shift_end(dt: datetime, shift_end: Optional[str]) -> datetime:
        """Return dt capped at shift_end time on the same day. No-op if shift_end is None."""
        if not shift_end:
            return dt
        try:
            eh, em = map(int, shift_end.split(":"))
            end_dt = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
            return min(dt, end_dt)
        except (ValueError, AttributeError):
            return dt

    def mark_attendance(
        self,
        employee_id: str,
        employee_name: str,
        camera_id: str,
        confidence: float,
        department: str = "",
        timestamp: Optional[datetime] = None,
        shift_start: Optional[str] = None,
        shift_end: Optional[str] = None,
    ) -> bool:
        """Mark attendance for today. Returns True if this is the first mark today.

        shift_start: "HH:MM" – arrivals after this time are marked late.
        shift_end:   "HH:MM" – last_seen is capped at this time (e.g. "17:00").
        """
        now = timestamp or datetime.now()
        today = now.date()
        capped = self._cap_to_shift_end(now, shift_end)

        is_late = False
        if shift_start:
            try:
                sh, sm = map(int, shift_start.split(":"))
                threshold = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
                is_late = now > threshold
            except (ValueError, AttributeError):
                pass

        with self._session() as session:
            existing = (
                session.query(AttendanceLog)
                .filter_by(employee_id=employee_id, date=today)
                .first()
            )
            if existing:
                existing.last_seen = capped
                if confidence > existing.confidence:
                    existing.confidence = round(confidence, 4)
                session.commit()
                return False

            entry = AttendanceLog(
                employee_id=employee_id,
                employee_name=employee_name,
                department=department,
                date=today,
                first_seen=now,
                last_seen=capped,
                camera_id=camera_id,
                confidence=round(confidence, 4),
                is_late=is_late,
            )
            session.add(entry)
            session.commit()
            logger.info(
                "Attendance marked: %s (%s) — %s",
                employee_name, employee_id,
                "LATE" if is_late else "on time",
            )
            return True

    def get_absent_employees(
        self,
        target_date: Optional[date] = None,
    ) -> List[Dict]:
        """Return all active employees who have NO attendance record for target_date."""
        target_date = target_date or datetime.now().date()
        with self._session() as session:
            all_emps = session.query(Employee).filter_by(is_active=True).all()
            attended_ids = {
                row.employee_id
                for row in session.query(AttendanceLog.employee_id).filter_by(
                    date=target_date
                ).all()
            }
            return [
                e.to_dict() for e in all_emps
                if e.employee_id not in attended_ids
            ]

    def get_attendance_by_date(
        self,
        target_date: Optional[date] = None,
    ) -> List[Dict]:
        """Return all attendance records for a given date (defaults to today)."""
        target_date = target_date or datetime.now().date()
        with self._session() as session:
            rows = (
                session.query(AttendanceLog)
                .filter_by(date=target_date)
                .order_by(AttendanceLog.first_seen)
                .all()
            )
            return [r.to_dict() for r in rows]

    def get_attendance_range(
        self,
        start: date,
        end: date,
        employee_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return attendance records between start and end dates (inclusive)."""
        with self._session() as session:
            q = session.query(AttendanceLog).filter(
                AttendanceLog.date >= start,
                AttendanceLog.date <= end,
            )
            if employee_id:
                q = q.filter_by(employee_id=employee_id)
            rows = q.order_by(AttendanceLog.date, AttendanceLog.first_seen).all()
            return [r.to_dict() for r in rows]

    def update_employee(
        self,
        employee_id: str,
        name: Optional[str] = None,
        department: Optional[str] = None,
    ) -> bool:
        """Update name and/or department for an existing employee. Returns False if not found."""
        with self._session() as session:
            emp = session.query(Employee).filter_by(employee_id=employee_id).first()
            if emp is None:
                return False
            if name is not None:
                emp.name = name
            if department is not None:
                emp.department = department
            session.commit()
            logger.info("Employee updated: %s", employee_id)
            return True

    def get_monthly_summary(self, year: int, month: int) -> List[Dict]:
        """Return per-employee attendance summary for the given year/month."""
        from calendar import monthrange
        start = date(year, month, 1)
        end = date(year, month, monthrange(year, month)[1])
        total_days = (end - start).days + 1

        with self._session() as session:
            all_emps = session.query(Employee).filter_by(is_active=True).all()
            att_records = (
                session.query(AttendanceLog)
                .filter(AttendanceLog.date >= start, AttendanceLog.date <= end)
                .all()
            )

        by_emp: Dict[str, list] = {}
        for r in att_records:
            by_emp.setdefault(r.employee_id, []).append(r)

        summary = []
        for emp in all_emps:
            records = by_emp.get(emp.employee_id, [])
            days_present = len(records)
            late_days = sum(1 for r in records if r.is_late)
            summary.append({
                "employee_id": emp.employee_id,
                "name": emp.name,
                "department": emp.department,
                "days_present": days_present,
                "days_absent": total_days - days_present,
                "late_days": late_days,
                "total_days": total_days,
                "attendance_rate": round(days_present / total_days * 100, 1) if total_days else 0.0,
            })

        return sorted(summary, key=lambda x: (x["department"], x["name"]))

    # ── VIP Visit Tracking ──────────────────────────────────

    def mark_vip_visit(
        self,
        employee_id: str,
        employee_name: str,
        camera_id: str,
        confidence: float,
        department: str = "",
        company_id: str = "",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Record VIP in/out times. Returns True if this is the first visit today (IN event)."""
        now = timestamp or datetime.now()
        today = now.date()

        with self._session() as session:
            existing = (
                session.query(VIPVisitLog)
                .filter_by(employee_id=employee_id, date=today)
                .first()
            )
            if existing:
                existing.out_time = now
                if confidence > existing.confidence:
                    existing.confidence = round(confidence, 4)
                session.commit()
                return False

            entry = VIPVisitLog(
                employee_id=employee_id,
                employee_name=employee_name,
                department=department,
                company_id=company_id,
                date=today,
                in_time=now,
                out_time=now,
                camera_id=camera_id,
                confidence=round(confidence, 4),
            )
            session.add(entry)
            session.commit()
            logger.info("VIP visit IN: %s (%s)", employee_name, employee_id)
            return True

    def get_vip_visits(
        self,
        target_date: Optional[date] = None,
        company_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return VIP visit records for a given date (defaults to today)."""
        target_date = target_date or datetime.now().date()
        with self._session() as session:
            q = session.query(VIPVisitLog).filter_by(date=target_date)
            if company_id:
                q = q.filter_by(company_id=company_id)
            rows = q.order_by(VIPVisitLog.in_time).all()
            return [r.to_dict() for r in rows]

    def get_vip_visits_range(
        self,
        start: date,
        end: date,
        employee_id: Optional[str] = None,
    ) -> List[Dict]:
        """Return VIP visit records between start and end dates (inclusive)."""
        with self._session() as session:
            q = session.query(VIPVisitLog).filter(
                VIPVisitLog.date >= start,
                VIPVisitLog.date <= end,
            )
            if employee_id:
                q = q.filter_by(employee_id=employee_id)
            rows = q.order_by(VIPVisitLog.date, VIPVisitLog.in_time).all()
            return [r.to_dict() for r in rows]

    def set_vip_status(self, employee_id: str, is_vip: bool) -> bool:
        """Set or clear VIP flag for an employee."""
        with self._session() as session:
            emp = session.query(Employee).filter_by(employee_id=employee_id).first()
            if emp is None:
                return False
            emp.is_vip = is_vip
            session.commit()
            logger.info("VIP status set to %s for %s", is_vip, employee_id)
            return True

    @classmethod
    def from_config(cls, cfg: Dict) -> "EmployeeDatabase":
        """Factory: build the DB URL from the config dict."""
        db_cfg = cfg.get("database", {})
        db_type = db_cfg.get("type", "sqlite")

        if db_type == "postgresql":
            pg = db_cfg.get("postgresql", {})
            url = (
                f"postgresql+psycopg2://{pg['user']}:{pg['password']}"
                f"@{pg['host']}:{pg['port']}/{pg['name']}"
            )
        else:
            path = db_cfg.get("sqlite", {}).get("path", "data/employees.db")
            url = f"sqlite:///{path}"

        return cls(database_url=url)
