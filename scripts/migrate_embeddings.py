"""
migrate_embeddings.py
─────────────────────────────────────────────────────────────
One-off migration: re-generate face embeddings for all employees
whose stored embedding dimension does not match the active model.

Reads each employee's photo_path, runs InsightFace detection,
and updates the database with the new 512-dim embedding.

Usage:
  python scripts/migrate_embeddings.py
  python scripts/migrate_embeddings.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate face embeddings to InsightFace 512-dim."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate even for employees who already have 512-dim embeddings.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from modules.employee_database import EmployeeDatabase
    from modules.face_detector import FaceDetector

    db = EmployeeDatabase.from_config(cfg)
    db.initialize()

    det_cfg = cfg.get("detection", {})
    detector = FaceDetector(
        yunet_model          = det_cfg.get("yunet_model", "data/models/face_detection_yunet_2023mar.onnx"),
        sface_model          = det_cfg.get("sface_model", "data/models/face_recognition_sface_2021dec.onnx"),
        det_thresh           = 0.45,
        nms_thresh           = 0.3,
        min_face_size        = 20,
        backend              = det_cfg.get("backend", "insightface"),
        insightface_model    = det_cfg.get("insightface_model", "buffalo_l"),
        insightface_root     = det_cfg.get("insightface_root", "data/models/insightface"),
        insightface_det_size = tuple(det_cfg.get("insightface_det_size", [640, 640])),
    )
    logger.info("Loading InsightFace model…")
    detector.initialize()
    target_dim = detector.EMBEDDING_DIM
    logger.info("Model ready — target embedding dim: %d", target_dim)

    employees = db.get_all_employees_with_embeddings()
    logger.info("Found %d active employees in database.", len(employees))

    updated = skipped = failed = 0

    for emp in employees:
        eid  = emp["employee_id"]
        name = emp["name"]

        # Check current embedding dim
        emb = emp.get("face_embedding")
        if emb is not None and isinstance(emb, np.ndarray):
            current_dim = emb.size
        else:
            current_dim = 0

        if current_dim == target_dim and not args.force:
            logger.info("[%s] %s — already %d-dim, skipping.", eid, name, target_dim)
            skipped += 1
            continue

        photo_path = emp.get("photo_path", "")
        if not photo_path or not Path(photo_path).exists():
            logger.warning("[%s] %s — photo not found at: %s", eid, name, photo_path)
            failed += 1
            continue

        img = cv2.imread(photo_path)
        if img is None:
            logger.warning("[%s] %s — could not read image: %s", eid, name, photo_path)
            failed += 1
            continue

        # Pass 1 — normal
        faces = detector.detect(img)
        if not faces:
            # Pass 2 — upscale 1.5×
            h, w = img.shape[:2]
            bigger = cv2.resize(img, (int(w * 1.5), int(h * 1.5)))
            faces = detector.detect(bigger)
        if not faces:
            # Pass 3 — CLAHE contrast
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            faces = detector.detect(enhanced)

        if not faces or faces[0].embedding is None:
            logger.warning("[%s] %s — no face detected in photo.", eid, name)
            failed += 1
            continue

        new_emb = faces[0].embedding.astype(np.float32)
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb /= norm

        if db.update_employee_embedding(eid, new_emb):
            logger.info("[%s] %s — updated %d→%d dim.", eid, name, current_dim, new_emb.size)
            updated += 1
        else:
            logger.error("[%s] %s — DB update failed.", eid, name)
            failed += 1

    logger.info("")
    logger.info("Migration complete — updated: %d | skipped: %d | failed: %d",
                updated, skipped, failed)
    if updated > 0:
        logger.info("Restart the app to reload the FAISS index with new embeddings.")


if __name__ == "__main__":
    main()
