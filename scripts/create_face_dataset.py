"""
create_face_dataset.py
─────────────────────────────────────────────────────────────
Utility to build a structured face image dataset by:

  1. Scanning a source directory of employee photos
     (expected layout: source_dir/<employee_id>/<image>.jpg)

  2. Detecting & aligning faces with InsightFace

  3. Saving aligned 112×112 crops to an output dataset directory

  4. Optionally augmenting each face (flip, brightness, blur)
     to improve recognition robustness.

Usage:
  python scripts/create_face_dataset.py \
      --source data/raw_photos \
      --output data/faces \
      --augment
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
from tqdm import tqdm

from modules.face_detector import FaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────

def augment_face(img: np.ndarray) -> list[np.ndarray]:
    """Return a list of augmented variants of *img* (112×112 BGR)."""
    variants = [img]

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Brightness +20%
    bright = np.clip(img.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    variants.append(bright)

    # Brightness -20%
    dark = np.clip(img.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    variants.append(dark)

    # Slight Gaussian blur (simulates low-res cameras)
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    variants.append(blurred)

    return variants


# ─────────────────────────────────────────────────────────────
# Per-employee processing
# ─────────────────────────────────────────────────────────────

def process_employee(
    employee_id: str,
    source_dir: Path,
    output_dir: Path,
    detector: FaceDetector,
    augment: bool = False,
) -> tuple[int, int]:
    """
    Process all photos for one employee.
    Returns (saved_count, failed_count).
    """
    emp_output = output_dir / employee_id
    emp_output.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not image_files:
        logger.warning("[%s] No images found in %s", employee_id, source_dir)
        return 0, 0

    saved, failed = 0, 0

    for idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("[%s] Unreadable: %s", employee_id, img_path.name)
            failed += 1
            continue

        faces = detector.detect(img)
        if not faces:
            logger.warning("[%s] No face in: %s", employee_id, img_path.name)
            failed += 1
            continue

        best = faces[0]

        # Prefer the pre-aligned crop from InsightFace if available
        face_img: np.ndarray
        if best.aligned_face is not None:
            face_img = cv2.resize(best.aligned_face, (112, 112))
        else:
            x1, y1, x2, y2 = best.bbox
            face_img = cv2.resize(img[y1:y2, x1:x2], (112, 112))

        images_to_save = augment_face(face_img) if augment else [face_img]

        for aug_idx, aug_img in enumerate(images_to_save):
            suffix = f"_aug{aug_idx}" if aug_idx > 0 else ""
            out_path = emp_output / f"{img_path.stem}{suffix}.jpg"
            cv2.imwrite(str(out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

    return saved, failed


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a face dataset from raw employee photos."
    )
    parser.add_argument(
        "--source", default="data/raw_photos",
        help="Root directory containing <employee_id>/<image> folders.",
    )
    parser.add_argument(
        "--output", default="data/faces",
        help="Output directory for aligned face crops.",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Generate augmented variants (flip, brightness, blur).",
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    source_root = Path(args.source)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        logger.error("Source directory not found: %s", source_root)
        sys.exit(1)

    # Each subdirectory = one employee
    employee_dirs = [d for d in source_root.iterdir() if d.is_dir()]
    if not employee_dirs:
        logger.error(
            "No employee sub-directories found in %s.\n"
            "Expected layout: %s/<employee_id>/<photo>.jpg",
            source_root, source_root,
        )
        sys.exit(1)

    logger.info(
        "Processing %d employees  augment=%s  source=%s → output=%s",
        len(employee_dirs), args.augment, source_root, output_root,
    )

    total_saved, total_failed = 0, 0
    for emp_dir in tqdm(employee_dirs, desc="Employees"):
        s, f = process_employee(
            employee_id=emp_dir.name,
            source_dir=emp_dir,
            output_dir=output_root,
            detector=detector,
            augment=args.augment,
        )
        total_saved += s
        total_failed += f

    logger.info(
        "Dataset creation complete – saved: %d images, failed: %d",
        total_saved, total_failed,
    )
    logger.info("Dataset saved to: %s", output_root.resolve())


if __name__ == "__main__":
    main()
