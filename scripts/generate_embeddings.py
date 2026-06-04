"""
generate_embeddings.py
─────────────────────────────────────────────────────────────
Batch-generates ArcFace embeddings for every employee in the
dataset directory and stores them in the database.

Run this after `create_face_dataset.py` to (re)build the
embeddings stored in the 'employees' table.

Usage:
  python scripts/generate_embeddings.py
  python scripts/generate_embeddings.py --faces-dir data/faces --update-all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from tqdm import tqdm

from modules.employee_database import EmployeeDatabase
from modules.face_embedding import FaceEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Compute the mean of multiple embeddings and re-normalise.
    Averaging multiple shots of the same person improves recognition
    stability compared to using a single embedding.
    """
    stacked = np.vstack(embeddings)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and store face embeddings for all employees."
    )
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument(
        "--faces-dir", default="data/faces",
        help="Root directory with <employee_id>/<image>.jpg sub-dirs.",
    )
    parser.add_argument(
        "--update-all", action="store_true",
        help="Re-generate embeddings even for employees who already have one.",
    )
    parser.add_argument(
        "--average", action="store_true",
        help="Average embeddings across all images for each employee "
             "(more robust than using a single image).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Setup database ──────────────────────────────────────
    db = EmployeeDatabase.from_config(cfg)
    db.initialize()

    # ── Setup embedding model ───────────────────────────────
    det_cfg = cfg.get("detection", {})
    use_gpu = cfg.get("performance", {}).get("use_gpu", False)
    embedder = FaceEmbedding(
        model_name=det_cfg.get("model", "buffalo_l"),
        ctx_id=0 if use_gpu else -1,
    )
    embedder.initialize()

    faces_root = Path(args.faces_dir)
    if not faces_root.exists():
        logger.error(
            "Faces directory not found: %s\n"
            "Run create_face_dataset.py first.",
            faces_root,
        )
        sys.exit(1)

    employees = db.get_all_employees()
    if not employees:
        logger.error("No employees found in database. Register employees first.")
        sys.exit(1)

    # Build quick-lookup set of employees that already have embeddings
    already_have = set()
    if not args.update_all:
        for emp in employees:
            if emp.get("has_embedding"):
                already_have.add(emp["employee_id"])

    logger.info(
        "Generating embeddings for %d employees  (update_all=%s, average=%s)",
        len(employees), args.update_all, args.average,
    )

    updated, skipped, failed = 0, 0, 0

    for emp in tqdm(employees, desc="Employees"):
        eid = emp["employee_id"]

        if eid in already_have:
            logger.debug("Skipping %s (already has embedding)", eid)
            skipped += 1
            continue

        emp_dir = faces_root / eid
        if not emp_dir.is_dir():
            logger.warning("[%s] No face directory at %s", eid, emp_dir)
            failed += 1
            continue

        # Collect all face images for this employee
        image_files = sorted([
            p for p in emp_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not image_files:
            logger.warning("[%s] No images in %s", eid, emp_dir)
            failed += 1
            continue

        embeddings: list[np.ndarray] = []

        for img_path in image_files:
            emb = embedder.get_embedding_from_file(str(img_path))
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.warning("[%s] Could not extract any embeddings", eid)
            failed += 1
            continue

        final_embedding = (
            average_embeddings(embeddings) if args.average and len(embeddings) > 1
            else embeddings[0]
        )

        success = db.update_employee_embedding(eid, final_embedding)
        if success:
            logger.debug("[%s] Embedding stored (%d images averaged)", eid, len(embeddings))
            updated += 1
        else:
            logger.error("[%s] Failed to update embedding in DB", eid)
            failed += 1

    logger.info(
        "Done – updated: %d, skipped: %d, failed: %d",
        updated, skipped, failed,
    )

    if updated > 0:
        logger.info(
            "Embeddings are stored in the database.\n"
            "Restart the recognition system to reload the FAISS index."
        )


if __name__ == "__main__":
    main()
