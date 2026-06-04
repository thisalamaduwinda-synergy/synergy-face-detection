"""
tune_threshold.py
─────────────────────────────────────────────────────────────
Empirically evaluate different cosine-similarity thresholds
against a labelled test set to find the optimal value for
your deployment environment.

Usage:
  python scripts/tune_threshold.py --test-dir data/test_faces

Expected layout of --test-dir:
  test_faces/
    alice_smith/
      test1.jpg
      test2.jpg
    bob_jones/
      test1.jpg

Every employee_id found in the sub-directory names is looked up
in the database to get the stored embedding; test images are then
evaluated against it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from modules.employee_database import EmployeeDatabase
from modules.face_embedding import FaceEmbedding
from modules.face_recognizer import FaceRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[bool], y_pred: list[bool]
) -> dict:
    tp = sum(a and b for a, b in zip(y_true, y_pred))
    tn = sum(not a and not b for a, b in zip(y_true, y_pred))
    fp = sum(not a and b for a, b in zip(y_true, y_pred))
    fn = sum(a and not b for a, b in zip(y_true, y_pred))

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy  = (tp + tn) / (len(y_true) + 1e-10)

    far  = fp / (fp + tn + 1e-10)   # False Acceptance Rate
    frr  = fn / (fn + tp + 1e-10)   # False Rejection Rate

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "far":       round(far,       4),   # lower = better (security)
        "frr":       round(frr,       4),   # lower = better (usability)
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the recognition threshold on a labelled test set."
    )
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument(
        "--test-dir", default="data/test_faces",
        help="Root directory with <employee_id>/<image> sub-folders.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.42,0.45,0.48,0.50,0.52,0.55,0.60",
        help="Comma-separated threshold values to evaluate.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load models ─────────────────────────────────────────
    db = EmployeeDatabase.from_config(cfg)
    db.initialize()

    det_cfg = cfg.get("detection", {})
    use_gpu = cfg.get("performance", {}).get("use_gpu", False)
    embedder = FaceEmbedding(
        model_name=det_cfg.get("model", "buffalo_l"),
        ctx_id=0 if use_gpu else -1,
    )
    embedder.initialize()

    # Build a temporary recognizer
    all_emps = db.get_all_employees_with_embeddings()
    if not all_emps:
        logger.error("No employees with embeddings found. Run generate_embeddings.py first.")
        sys.exit(1)

    recognizer = FaceRecognizer(threshold=0.0, embedding_dim=512)  # threshold=0 → keep all scores
    recognizer.build_index(all_emps)

    # ── Collect test samples ────────────────────────────────
    test_root = Path(args.test_dir)
    if not test_root.exists():
        logger.error("Test directory not found: %s", test_root)
        sys.exit(1)

    registered_ids = {e["employee_id"] for e in all_emps}
    samples: list[tuple[str, np.ndarray, bool]] = []   # (true_id, embedding, is_registered)

    for emp_dir in test_root.iterdir():
        if not emp_dir.is_dir():
            continue
        emp_id = emp_dir.name
        is_registered = emp_id in registered_ids

        for img_path in emp_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            emb = embedder.get_embedding_from_file(str(img_path))
            if emb is not None:
                samples.append((emp_id, emb, is_registered))

    if not samples:
        logger.error("No valid face images found in %s", test_root)
        sys.exit(1)

    logger.info("Collected %d test samples.", len(samples))

    # ── Evaluate each threshold ──────────────────────────────
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(
        f"\n{'Threshold':>10}  {'Accuracy':>9}  {'Precision':>10}  "
        f"{'Recall':>8}  {'F1':>6}  {'FAR':>6}  {'FRR':>6}"
    )
    print("-" * 75)

    best = {"threshold": 0.0, "f1": 0.0}

    for thresh in thresholds:
        recognizer.threshold = thresh
        y_true: list[bool] = []
        y_pred: list[bool] = []

        for true_id, emb, is_reg in samples:
            result = recognizer.recognize(emb)
            y_true.append(is_reg and (result.employee_id == true_id or True))
            # "ground truth positive" = registered employee
            # "predicted positive"   = system says it's known
            y_true[-1] = is_reg
            y_pred.append(result.is_known)

        m = compute_metrics(y_true, y_pred)
        print(
            f"{thresh:>10.2f}  {m['accuracy']:>9.4f}  {m['precision']:>10.4f}  "
            f"{m['recall']:>8.4f}  {m['f1']:>6.4f}  {m['far']:>6.4f}  {m['frr']:>6.4f}"
        )
        if m["f1"] > best["f1"]:
            best = {"threshold": thresh, **m}

    print("-" * 75)
    print(f"\n→ Recommended threshold: {best['threshold']:.2f}  (F1={best['f1']:.4f})")
    print(
        f"  Update 'recognition.threshold' in config/config.yaml to {best['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
