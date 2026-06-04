"""
face_embedding.py
─────────────────────────────────────────────────────────────
Utilities for face embedding management using OpenCV SFace:
  • Standalone embedding extraction (for registration scripts)
  • Batch embedding generation
  • Embedding normalisation / distance helpers
  • Serialisation helpers (numpy ↔ bytes for DB storage)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_YUNET = "data/models/face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE = "data/models/face_recognition_sface_2021dec.onnx"


# ─────────────────────────────────────────────────────────────
# Embedding model wrapper
# ─────────────────────────────────────────────────────────────

class FaceEmbedding:
    """
    Extracts 128-dimensional SFace embeddings from face images.

    Kept separate from FaceDetector so batch scripts can generate
    embeddings without running the real-time detection loop.
    """

    EMBEDDING_DIM = 128

    def __init__(
        self,
        yunet_model: str = DEFAULT_YUNET,
        sface_model: str = DEFAULT_SFACE,
    ) -> None:
        self.yunet_model = str(yunet_model)
        self.sface_model = str(sface_model)
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._current_input_size: Tuple[int, int] = (0, 0)

    def initialize(self) -> None:
        if self._detector is not None:
            return
        self._detector = cv2.FaceDetectorYN.create(
            model=self.yunet_model,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_model,
            config="",
        )
        logger.info("FaceEmbedding (SFace) models loaded.")

    # ── Single image ────────────────────────────────────────

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Return the L2-normalised 128-d embedding for the largest face
        in *image* (BGR, uint8).  Returns None if no face detected.
        """
        if self._detector is None:
            self.initialize()
        assert self._detector is not None
        assert self._recognizer is not None

        h, w = image.shape[:2]
        if (w, h) != self._current_input_size:
            self._detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        best = max(faces, key=lambda f: float(f[14]))
        try:
            aligned = self._recognizer.alignCrop(image, best)
            feat = self._recognizer.feature(aligned)
            emb = np.array(feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as exc:
            logger.warning("Embedding extraction failed: %s", exc)
            return None

    def get_embedding_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from disk and return its face embedding."""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("Cannot read image: %s", image_path)
            return None
        return self.get_embedding(img)

    # ── Batch processing ────────────────────────────────────

    def get_embeddings_batch(
        self, images: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """Extract embeddings for a list of images."""
        return [self.get_embedding(img) for img in images]

    def get_embeddings_from_directory(
        self, directory: str, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ) -> dict:
        """
        Process all images in *directory*.
        Returns {filename_stem: embedding} for every file where a face
        was successfully detected.
        """
        results = {}
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error("Directory not found: %s", directory)
            return results

        image_paths = [
            p for p in dir_path.iterdir()
            if p.suffix.lower() in extensions
        ]
        logger.info("Processing %d images in %s...", len(image_paths), directory)

        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Skipping unreadable file: %s", path.name)
                continue
            emb = self.get_embedding(img)
            if emb is not None:
                results[path.stem] = emb
            else:
                logger.warning("No face detected in: %s", path.name)

        logger.info("Extracted %d embeddings from %d images", len(results), len(image_paths))
        return results

    # ── Similarity helpers ──────────────────────────────────

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        """Return L2-normalised vector; return input unchanged for zero norm."""
        vec = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return vec.copy()
        return vec / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised embedding vectors."""
        a_n = FaceEmbedding._normalise(a)
        b_n = FaceEmbedding._normalise(b)
        return float(np.dot(a_n, b_n))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """L2 distance between two embedding vectors."""
        return float(np.linalg.norm(a - b))

    # ── Serialisation ───────────────────────────────────────

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Serialise a float32 numpy embedding to raw bytes for DB storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(data: bytes, dim: int = 128) -> np.ndarray:
        """Deserialise bytes back to a float32 numpy array."""
        emb = np.frombuffer(data, dtype=np.float32)
        if emb.shape[0] != dim:
            raise ValueError(
                f"Expected {dim}-d embedding, got {emb.shape[0]} values."
            )
        return emb.copy()

    @property
    def is_ready(self) -> bool:
        return self._detector is not None
