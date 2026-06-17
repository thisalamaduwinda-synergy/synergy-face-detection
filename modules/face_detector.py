"""
face_detector.py
─────────────────────────────────────────────────────────────
Face detection + embedding extraction with two backends:

    1) InsightFace (primary): SCRFD detector + ArcFace embedding
    2) OpenCV fallback: YuNet detector + SFace embedding

The class preserves the same public API used by the rest of the project.
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    """One detected face in a frame."""
    # Bounding box in (x1, y1, x2, y2) format
    bbox: Tuple[int, int, int, int]
    # Detection confidence [0, 1]
    det_score: float
    # 5-point facial landmarks (optional) — shape (5, 2)
    kps: Optional[np.ndarray] = None
    # 128-d SFace embedding (L2-normalised)
    embedding: Optional[np.ndarray] = None
    # Aligned/cropped face image (112×112 BGR)
    aligned_face: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


# ─────────────────────────────────────────────────────────────
# Face detector
# ─────────────────────────────────────────────────────────────

class FaceDetector:
    """
        Detects faces and extracts embeddings.

        Backend modes
        -------------
        insightface (default)
            SCRFD + ArcFace model pack (buffalo_l)

        opencv
            YuNet + SFace

    Parameters
    ----------
    yunet_model : str
        Path to face_detection_yunet_2023mar.onnx
    sface_model : str
        Path to face_recognition_sface_2021dec.onnx
    det_thresh : float
        Minimum detection score to keep a face [0, 1].
    nms_thresh : float
        NMS IOU threshold.
    min_face_size : int
        Discard faces smaller than this many pixels in height.
    """

    EMBEDDING_DIM = 512

    def __init__(
        self,
        yunet_model: str = "data/models/face_detection_yunet_2023mar.onnx",
        sface_model: str = "data/models/face_recognition_sface_2021dec.onnx",
        det_thresh: float = 0.6,
        nms_thresh: float = 0.3,
        min_face_size: int = 40,
        backend: str = "insightface",
        insightface_model: str = "buffalo_l",
        insightface_root: str = "data/models/insightface",
        insightface_det_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = False,
    ) -> None:
        self.yunet_model = str(yunet_model)
        self.sface_model = str(sface_model)
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.min_face_size = min_face_size
        self.backend = (backend or "insightface").lower()
        self.insightface_model = insightface_model
        self.insightface_root = str(insightface_root)
        self.insightface_det_size = tuple(insightface_det_size)
        self.use_gpu = use_gpu

        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._if_app = None
        self._current_input_size: Tuple[int, int] = (0, 0)
        self._initialised = False

    # ── Initialisation ──────────────────────────────────────

    def initialize(self) -> None:
        """Load model weights from disk."""
        if self._initialised:
            return

        if self.backend == "insightface":
            if self._initialize_insightface():
                self._initialised = True
                self.EMBEDDING_DIM = 512
                return
            logger.warning("InsightFace init failed. Falling back to OpenCV YuNet+SFace.")
            self.backend = "opencv"

        self._initialize_opencv()
        self._initialised = True
        self.EMBEDDING_DIM = 128

    def _initialize_opencv(self) -> None:
        """Initialise OpenCV YuNet + SFace backend."""

        for path in (self.yunet_model, self.sface_model):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Download it by running:  python scripts/download_models.py"
                )

        logger.info("Loading YuNet face detector + SFace recognizer…")
        t0 = time.perf_counter()

        # Initial size; updated per-frame in detect() via setInputSize
        self._detector = cv2.FaceDetectorYN.create(
            model=self.yunet_model,
            config="",
            input_size=(320, 320),
            score_threshold=self.det_thresh,
            nms_threshold=self.nms_thresh,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_model,
            config="",
        )

        elapsed = time.perf_counter() - t0
        logger.info("YuNet + SFace models ready (%.2fs)", elapsed)

    def _initialize_insightface(self) -> bool:
        """Initialise InsightFace SCRFD + ArcFace backend."""
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as exc:
            logger.error("insightface import failed: %s", exc)
            return False

        root_dir = Path(self.insightface_root)
        pack_dir = root_dir / self.insightface_model
        if not pack_dir.exists() or not list(pack_dir.glob("*.onnx")):
            self._download_insightface_pack(self.insightface_model, root_dir)

        logger.info(
            "Loading InsightFace model pack '%s' from %s",
            self.insightface_model,
            root_dir,
        )
        t0 = time.perf_counter()

        ctx_id = 0 if self.use_gpu else -1
        try:
            self._if_app = FaceAnalysis(
                name=self.insightface_model,
                root=str(root_dir),
            )
            self._if_app.prepare(
                ctx_id=ctx_id,
                det_thresh=float(self.det_thresh),
                det_size=self.insightface_det_size,
            )
            elapsed = time.perf_counter() - t0
            logger.info("InsightFace ready (%.2fs) [%s]", elapsed, "GPU" if self.use_gpu else "CPU")
            return True
        except Exception as exc:
            logger.warning("InsightFace initialisation failed: %s", exc)

            # Legacy insightface versions may fail on model packs containing
            # unsupported task ONNX files (e.g., 2d106det/genderage).
            compat_model_name = f"{self.insightface_model}_compat"
            try:
                self._build_insightface_compat_pack(
                    source_dir=root_dir / self.insightface_model,
                    target_dir=root_dir / compat_model_name,
                )
                self._if_app = FaceAnalysis(
                    name=compat_model_name,
                    root=str(root_dir),
                )
                self._if_app.prepare(
                    ctx_id=ctx_id,
                    det_thresh=float(self.det_thresh),
                    det_size=self.insightface_det_size,
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "InsightFace ready via compatibility pack '%s' (%.2fs)",
                    compat_model_name,
                    elapsed,
                )
                return True
            except Exception as compat_exc:
                logger.error("InsightFace compatibility pack failed: %s", compat_exc)
                self._if_app = None
                return False

    @staticmethod
    def _build_insightface_compat_pack(source_dir: Path, target_dir: Path) -> None:
        """
        Build a compatibility pack that keeps only ONNX files routable by
        legacy insightface.model_zoo routers.
        """
        if not source_dir.exists():
            raise FileNotFoundError(f"InsightFace pack not found: {source_dir}")

        try:
            from insightface.model_zoo import get_model  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Cannot import insightface.model_zoo: {exc}")

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        for onnx in sorted(source_dir.glob("*.onnx")):
            try:
                model = get_model(str(onnx))
                taskname = getattr(model, "taskname", "")
                if taskname in {"detection", "recognition"}:
                    shutil.copy2(onnx, target_dir / onnx.name)
                    kept += 1
            except Exception:
                continue

        if kept < 2:
            raise RuntimeError(
                f"Compatibility pack incomplete (kept {kept} model(s)) from {source_dir}"
            )

    @staticmethod
    def _download_insightface_pack(model_name: str, root_dir: Path) -> None:
        """
        Download and extract an InsightFace model pack (e.g. buffalo_l).
        """
        root_dir.mkdir(parents=True, exist_ok=True)
        pack_dir = root_dir / model_name
        if pack_dir.exists() and list(pack_dir.glob("*.onnx")):
            return

        url = f"https://github.com/deepinsight/insightface/releases/download/v0.7/{model_name}.zip"
        zip_path = root_dir / f"{model_name}.zip"
        temp_extract = root_dir / f".{model_name}_extract"

        logger.info("Downloading InsightFace model pack: %s", url)
        urllib.request.urlretrieve(url, str(zip_path))

        if temp_extract.exists():
            shutil.rmtree(temp_extract, ignore_errors=True)
        temp_extract.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_extract)

        candidate = temp_extract / model_name
        if not candidate.exists():
            # Some archives may extract without top-level folder.
            candidate = temp_extract

        if pack_dir.exists():
            shutil.rmtree(pack_dir, ignore_errors=True)
        pack_dir.mkdir(parents=True, exist_ok=True)

        for onnx in candidate.glob("*.onnx"):
            shutil.copy2(onnx, pack_dir / onnx.name)

        shutil.rmtree(temp_extract, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink()

        if not list(pack_dir.glob("*.onnx")):
            raise RuntimeError(f"No ONNX files found in downloaded pack: {model_name}")

    # ── Detection ───────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Run detection + embedding on *frame* (BGR uint8).
        Returns a list of DetectedFace objects, sorted by area (largest first).
        """
        if not self._initialised:
            self.initialize()

        if frame is None or frame.size == 0:
            return []

        if self.backend == "insightface" and self._if_app is not None:
            return self._detect_insightface(frame)

        return self._detect_opencv(frame)

    def _detect_opencv(self, frame: np.ndarray) -> List[DetectedFace]:
        h, w = frame.shape[:2]
        if (w, h) != self._current_input_size:
            self._detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        try:
            _, faces_raw = self._detector.detect(frame)
        except Exception as exc:
            logger.error("Detection error: %s", exc)
            return []

        if faces_raw is None or len(faces_raw) == 0:
            return []

        results: List[DetectedFace] = []
        for face_row in faces_raw:
            x, y = int(face_row[0]), int(face_row[1])
            fw, fh = int(face_row[2]), int(face_row[3])
            score = float(face_row[14])

            if score < self.det_thresh:
                continue
            if fh < self.min_face_size or fw < self.min_face_size:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            kps = face_row[4:14].reshape(5, 2).astype(np.float32)

            aligned: Optional[np.ndarray] = None
            embedding: Optional[np.ndarray] = None
            try:
                aligned = self._recognizer.alignCrop(frame, face_row)
                feat = self._recognizer.feature(aligned)
                emb = np.array(feat, dtype=np.float32).flatten()
                norm = np.linalg.norm(emb)
                embedding = emb / norm if norm > 0 else emb
            except Exception as exc:
                logger.warning("Embedding error: %s", exc)

            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=score,
                    kps=kps,
                    embedding=embedding,
                    aligned_face=aligned,
                )
            )

        results.sort(key=lambda d: d.area, reverse=True)
        return results

    def _detect_insightface(self, frame: np.ndarray) -> List[DetectedFace]:
        h, w = frame.shape[:2]
        try:
            faces = self._if_app.get(frame)
        except Exception as exc:
            logger.error("InsightFace detection error: %s", exc)
            return []

        if not faces:
            return []

        results: List[DetectedFace] = []
        for f in faces:
            bbox = np.array(f.bbox).astype(np.int32).tolist()
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))

            fw = x2 - x1
            fh = y2 - y1
            if fw < self.min_face_size or fh < self.min_face_size:
                continue

            score = float(getattr(f, "det_score", 0.0) or 0.0)
            if score < self.det_thresh:
                continue

            kps = getattr(f, "kps", None)
            if kps is not None:
                kps = np.array(kps, dtype=np.float32)

            emb = getattr(f, "normed_embedding", None)
            embedding: Optional[np.ndarray] = None
            if emb is not None:
                embedding = np.array(emb, dtype=np.float32).flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=score,
                    kps=kps,
                    embedding=embedding,
                    aligned_face=None,
                )
            )

        # Sort: largest face first (most prominent person)
        results.sort(key=lambda d: d.area, reverse=True)
        return results

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[DetectedFace]]:
        """Run detection on a list of frames sequentially."""
        return [self.detect(f) for f in frames]

    # ── Drawing utilities ───────────────────────────────────

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        faces: List[DetectedFace],
        labels: Optional[List[str]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes (and optional labels) on *frame*.
        Returns an annotated copy.
        """
        out = frame.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = labels[i] if labels and i < len(labels) else ""
            if label:
                # Dark background behind text for readability
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    out,
                    (x1, y1 - th - baseline - 4),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    out,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Draw landmarks
            if face.kps is not None:
                for kp in face.kps:
                    cv2.circle(out, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

        return out

    @property
    def is_ready(self) -> bool:
        return self._initialised
