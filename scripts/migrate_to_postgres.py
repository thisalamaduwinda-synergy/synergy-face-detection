"""
migrate_to_postgres.py
──────────────────────────────────────────────────────────────
Migrate all data from SQLite  →  PostgreSQL.

Usage:
    python scripts/migrate_to_postgres.py

Prerequisites:
    1. PostgreSQL installed and running
    2. Database + user created (see instructions printed below)
    3. DB_PASSWORD set in .env  (or passed via --password)

What it migrates:
    • employees          (including face embeddings)
    • detection_logs
    • attendance_logs
    • vip_visit_logs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── resolve project root ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sqlite_url(cfg: dict) -> str:
    path = cfg["database"]["sqlite"]["path"]
    if not Path(path).is_absolute():
        path = str(ROOT / path)
    return f"sqlite:///{path}"


def _pg_url(cfg: dict, password: str) -> str:
    pg = cfg["database"]["postgresql"]
    return (
        f"postgresql+psycopg2://{pg['user']}:{password}"
        f"@{pg['host']}:{pg['port']}/{pg['name']}"
    )


def _count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


# ─────────────────────────────────────────────────────────────
# Main migration
# ─────────────────────────────────────────────────────────────

def migrate(pg_password: str) -> None:
    cfg = _load_cfg()

    # ── Connect to both databases ──────────────────────────
    sqlite_engine = create_engine(
        _sqlite_url(cfg),
        connect_args={"check_same_thread": False},
    )
    pg_engine = create_engine(
        _pg_url(cfg, pg_password),
        pool_pre_ping=True,
    )

    print("\n Connecting to PostgreSQL…", end=" ")
    try:
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("OK")
    except Exception as exc:
        print(f"FAILED\n\nError: {exc}")
        print("\n── Fix checklist ───────────────────────────────────")
        print("  1. Is PostgreSQL running?  (check Services in Task Manager)")
        print("  2. Is the database created?  Run in psql:")
        print("       CREATE DATABASE face_recognition;")
        print("       CREATE USER postgres WITH PASSWORD 'your_password';")
        print("       GRANT ALL ON DATABASE face_recognition TO postgres;")
        print("  3. Is DB_PASSWORD correct in .env?")
        sys.exit(1)

    # ── Create tables in PostgreSQL via SQLAlchemy ORM ────
    print(" Creating tables in PostgreSQL…", end=" ")
    from modules.employee_database import Base
    Base.metadata.create_all(pg_engine)
    print("OK")

    SqliteSession = sessionmaker(bind=sqlite_engine)
    PgSession     = sessionmaker(bind=pg_engine)

    # ── Migrate employees ──────────────────────────────────
    print("\n Migrating employees…")
    with SqliteSession() as src, PgSession() as dst:
        rows = src.execute(text("SELECT * FROM employees")).fetchall()
        cols = src.execute(text("SELECT * FROM employees")).keys()

        inserted = skipped = 0
        for row in rows:
            data = dict(zip(cols, row))
            exists = dst.execute(
                text("SELECT 1 FROM employees WHERE employee_id = :eid"),
                {"eid": data["employee_id"]},
            ).fetchone()
            if exists:
                skipped += 1
                continue
            dst.execute(
                text("""
                    INSERT INTO employees
                        (employee_id, name, department, company_id, photo_path,
                         face_embedding, registered_at, is_active, is_vip)
                    VALUES
                        (:employee_id, :name, :department, :company_id, :photo_path,
                         :face_embedding, :registered_at, :is_active, :is_vip)
                """),
                data,
            )
            inserted += 1
        dst.commit()
    print(f"   ✓ {inserted} inserted,  {skipped} already existed")

    # ── Migrate detection_logs ─────────────────────────────
    print(" Migrating detection_logs…")
    with SqliteSession() as src, PgSession() as dst:
        rows = src.execute(text("SELECT * FROM detection_logs")).fetchall()
        keys = src.execute(text("SELECT * FROM detection_logs")).keys()

        # Clear existing in PG to avoid duplicates on re-run
        existing = _count(pg_engine, "detection_logs")
        if existing > 0:
            print(f"   PostgreSQL already has {existing} rows — clearing before re-import…")
            dst.execute(text("DELETE FROM detection_logs"))
            dst.commit()

        batch = []
        for row in rows:
            batch.append(dict(zip(keys, row)))
            if len(batch) == 500:
                dst.execute(
                    text("""
                        INSERT INTO detection_logs
                            (id, timestamp, camera_id, employee_id, employee_name,
                             confidence, is_known, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                             frame_path)
                        VALUES
                            (:id, :timestamp, :camera_id, :employee_id, :employee_name,
                             :confidence, :is_known, :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                             :frame_path)
                    """),
                    batch,
                )
                dst.commit()
                batch = []
        if batch:
            dst.execute(
                text("""
                    INSERT INTO detection_logs
                        (id, timestamp, camera_id, employee_id, employee_name,
                         confidence, is_known, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                         frame_path)
                    VALUES
                        (:id, :timestamp, :camera_id, :employee_id, :employee_name,
                         :confidence, :is_known, :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                         :frame_path)
                """),
                batch,
            )
            dst.commit()
    print(f"   ✓ {len(rows)} detection logs migrated")

    # ── Migrate attendance_logs ────────────────────────────
    print(" Migrating attendance_logs…")
    with SqliteSession() as src, PgSession() as dst:
        rows = src.execute(text("SELECT * FROM attendance_logs")).fetchall()
        keys = src.execute(text("SELECT * FROM attendance_logs")).keys()

        existing = _count(pg_engine, "attendance_logs")
        if existing > 0:
            print(f"   PostgreSQL already has {existing} rows — clearing before re-import…")
            dst.execute(text("DELETE FROM attendance_logs"))
            dst.commit()

        for row in rows:
            data = dict(zip(keys, row))
            dst.execute(
                text("""
                    INSERT INTO attendance_logs
                        (id, employee_id, employee_name, department, date,
                         first_seen, last_seen, camera_id, confidence, is_late)
                    VALUES
                        (:id, :employee_id, :employee_name, :department, :date,
                         :first_seen, :last_seen, :camera_id, :confidence, :is_late)
                """),
                data,
            )
        dst.commit()
    print(f"   ✓ {len(rows)} attendance records migrated")

    # ── Migrate vip_visit_logs ─────────────────────────────
    print(" Migrating vip_visit_logs…")
    with SqliteSession() as src, PgSession() as dst:
        rows = src.execute(text("SELECT * FROM vip_visit_logs")).fetchall()
        keys = src.execute(text("SELECT * FROM vip_visit_logs")).keys()

        existing = _count(pg_engine, "vip_visit_logs")
        if existing > 0:
            dst.execute(text("DELETE FROM vip_visit_logs"))
            dst.commit()

        for row in rows:
            data = dict(zip(keys, row))
            dst.execute(
                text("""
                    INSERT INTO vip_visit_logs
                        (id, employee_id, employee_name, department, company_id,
                         date, in_time, out_time, camera_id, confidence)
                    VALUES
                        (:id, :employee_id, :employee_name, :department, :company_id,
                         :date, :in_time, :out_time, :camera_id, :confidence)
                """),
                data,
            )
        dst.commit()
    print(f"   ✓ {len(rows)} VIP visit records migrated")

    # ── Final counts ───────────────────────────────────────
    print("\n── Verification ────────────────────────────────────")
    for table in ("employees", "detection_logs", "attendance_logs", "vip_visit_logs"):
        n = _count(pg_engine, table)
        print(f"   {table:25s}: {n} rows")

    # ── Switch config to PostgreSQL ────────────────────────
    print("\n Updating config.yaml  →  type: postgresql …", end=" ")
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["database"]["type"] = "postgresql"
    raw["database"]["postgresql"]["password"] = ""   # keep password in .env only
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)
    print("OK")

    print("\n Migration complete!")
    print(" Restart the app — it will now use PostgreSQL.\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    parser.add_argument(
        "--password",
        default=os.environ.get("DB_PASSWORD", ""),
        help="PostgreSQL password (default: DB_PASSWORD from .env)",
    )
    args = parser.parse_args()

    if not args.password:
        print("\n No PostgreSQL password provided.")
        print(" Set DB_PASSWORD in .env  or  pass  --password yourpass\n")
        sys.exit(1)

    migrate(args.password)
