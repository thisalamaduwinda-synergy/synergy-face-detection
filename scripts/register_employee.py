"""
register_employee.py
─────────────────────────────────────────────────────────────
Interactive CLI to register a new employee into the database.

Usage (single employee):
  python scripts/register_employee.py \
      --id EMP001 --name "Alice Smith" --dept "Engineering" \
      --photo data/employees/alice.jpg

Usage (batch from CSV):
  python scripts/register_employee.py --csv employees.csv

CSV format (no header row expected – or use --has-header):
  employee_id,name,department,photo_path
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

from modules.employee_database import EmployeeDatabase
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_and_embed(photo_path: str, detector: FaceDetector) -> np.ndarray | None:
    """Return the L2-normalised embedding for the largest face in *photo_path*."""
    img = cv2.imread(photo_path)
    if img is None:
        logger.error("Cannot read photo: %s", photo_path)
        return None

    faces = detector.detect(img)
    if not faces:
        logger.warning("No face detected in: %s", photo_path)
        return None

    best = faces[0]  # Already sorted by size (largest first)
    if best.embedding is None:
        logger.warning("Embedding extraction failed for: %s", photo_path)
        return None

    logger.info(
        "Face detected (det_score=%.2f, size=%dx%d)",
        best.det_score, best.width, best.height,
    )
    return best.embedding


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average multiple embeddings and return a single L2-normalised vector."""
    if not embeddings:
        raise ValueError("No embeddings to average")
    stacked = np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings])
    avg = stacked.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg /= norm
    return avg


def register_one(
    db: EmployeeDatabase,
    detector: FaceDetector,
    employee_id: str,
    name: str,
    department: str,
    photo_paths: list[str],
) -> bool:
    """Register a single employee. Returns True on success."""
    if len(photo_paths) < 1:
        logger.error("At least 1 photo is required.")
        return False
    if len(photo_paths) > 5:
        logger.error("A maximum of 5 photos is allowed.")
        return False

    # Validate that all photos exist
    for photo_path in photo_paths:
        if not Path(photo_path).is_file():
            logger.error("Photo not found: %s", photo_path)
            return False

    embeddings: list[np.ndarray] = []
    for photo_path in photo_paths:
        emb = detect_and_embed(photo_path, detector)
        if emb is None:
            logger.error("Skipping %s – could not generate embedding from %s.", employee_id, photo_path)
            return False
        embeddings.append(emb)

    embedding = average_embeddings(embeddings)

    first_photo_path = str(Path(photo_paths[0]).resolve())
    logger.info("Using %d photos for %s", len(photo_paths), employee_id)

    if embedding is None:
        logger.error("Skipping %s – could not generate embedding.", employee_id)
        return False

    try:
        db.add_employee(
            employee_id=employee_id,
            name=name,
            department=department,
            embedding=embedding,
            photo_path=first_photo_path,
        )
        logger.info("Registered: %s (%s) – dept: %s", name, employee_id, department)
        return True
    except ValueError as e:
        logger.warning("Skipping: %s", e)
        return False
    except Exception as e:
        logger.error("DB error for %s: %s", employee_id, e)
        return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register employees into the face recognition database."
    )
    parser.add_argument("--config", default="config/config.yaml")

    # Single-employee mode
    single = parser.add_argument_group("single employee")
    single.add_argument("--id",     dest="employee_id", help="Employee ID")
    single.add_argument("--name",   dest="name",        help="Full name")
    single.add_argument("--dept",   dest="department",  default="",   help="Department")
    single.add_argument(
        "--photo",
        dest="photos",
        action="append",
        default=[],
        help="Path to a clear face photo (repeat flag for multiple photos, max 5)",
    )
    single.add_argument(
        "--photos",
        dest="photos_list",
        nargs="+",
        default=[],
        help="Space-separated photo paths (1-5)",
    )

    # Batch mode
    batch = parser.add_argument_group("batch (CSV)")
    batch.add_argument("--csv",        dest="csv_file",   help="Path to CSV file")
    batch.add_argument("--has-header", dest="has_header", action="store_true",
                       help="Skip first row (header)")

    # Webcam capture mode
    webcam = parser.add_argument_group("webcam capture")
    webcam.add_argument(
        "--capture", action="store_true",
        help="Capture face from webcam instead of loading a photo file",
    )
    webcam.add_argument("--cam-id", type=int, default=0, help="Webcam device index")

    args = parser.parse_args()

    cfg = load_config(args.config)
    db_cfg = cfg.get("database", {})
    db_type = db_cfg.get("type", "sqlite")

    if db_type == "postgresql":
        pg = db_cfg.get("postgresql", {})
        db_url = (
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}"
            f"@{pg['host']}:{pg['port']}/{pg['name']}"
        )
    else:
        path = db_cfg.get("sqlite", {}).get("path", "data/employees.db")
        db_url = f"sqlite:///{path}"

    db = EmployeeDatabase(db_url)
    db.initialize()

    det_cfg = cfg.get("detection", {})
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

    # ── Webcam capture ──────────────────────────────────────
    if args.capture:
        if not args.employee_id or not args.name:
            parser.error("--id and --name are required with --capture")

        logger.info("Opening webcam %d …  Press SPACE to capture, Q to quit.", args.cam_id)
        cap = cv2.VideoCapture(args.cam_id)
        photo_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Cannot read from webcam.")
                break

            cv2.imshow("Capture – SPACE to take photo, Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                save_dir = Path("data/employees")
                save_dir.mkdir(parents=True, exist_ok=True)
                safe_id = "".join(c for c in args.employee_id if c.isalnum() or c in "-_")
                photo_path = str(save_dir / f"{safe_id}_capture.jpg")
                cv2.imwrite(photo_path, frame)
                logger.info("Photo saved: %s", photo_path)
                break
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if photo_path:
            register_one(db, detector, args.employee_id, args.name,
                         args.department or "", [photo_path])
        return

    # ── Single employee ─────────────────────────────────────
    if args.employee_id and args.name:
        photo_paths = [*args.photos, *args.photos_list]
        if not photo_paths:
            parser.error("At least one --photo/--photos value is required for single registration")
        register_one(db, detector, args.employee_id, args.name,
                     args.department or "", photo_paths)

    # ── Batch CSV ───────────────────────────────────────────
    elif args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.is_file():
            logger.error("CSV file not found: %s", csv_path)
            sys.exit(1)

        success, skipped = 0, 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if args.has_header:
                next(reader, None)
            for row in reader:
                if len(row) < 4:
                    logger.warning("Invalid CSV row (need 4 cols): %s", row)
                    skipped += 1
                    continue
                emp_id, name, dept, photo = (c.strip() for c in row[:4])
                ok = register_one(db, detector, emp_id, name, dept, [photo])
                if ok:
                    success += 1
                else:
                    skipped += 1

        logger.info("Batch complete – registered: %d, skipped: %d", success, skipped)

    else:
        parser.print_help()

    # Report final count
    logger.info("Total active employees in DB: %d", db.employee_count())


if __name__ == "__main__":
    main()
