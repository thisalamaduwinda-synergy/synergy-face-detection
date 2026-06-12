"""
face_recognizer.py
─────────────────────────────────────────────────────────────
FAISS-backed face recognition engine.

Flow:
  1. Receives a 128-d L2-normalised embedding from FaceDetector.
  2. Queries the in-memory FAISS index (IndexFlatIP → cosine similarity).
  3. If best match score ≥ threshold → return employee info.
  4. Otherwise → mark as UNKNOWN.

The index is rebuilt from the database on startup and whenever an
employee is added or removed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss as faiss   # noqa: PLC0414  (re-bind so type-checker sees it as bound)
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed – recognition unavailable.")


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

UNKNOWN_LABEL = "Unknown"
UNKNOWN_ID = "UNKNOWN"


@dataclass
class RecognitionResult:
    """Result of matching an embedding against the employee database."""
    employee_id: str
    name: str
    department: str
    confidence: float                   # cosine similarity [0, 1]
    is_known: bool
    latency_ms: float = 0.0
    # Bounding box copied from detection for downstream use
    bbox: Optional[Tuple[int, int, int, int]] = None


# ─────────────────────────────────────────────────────────────
# Face Recognizer
# ─────────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Maintains a FAISS index and maps embedding positions to employee records.

    Thread-safe: uses a lock around index reads/writes.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity score to consider a face a positive match.
        Typical range: 0.55 – 0.80.  Tune with scripts/tune_threshold.py.
    embedding_dim : int
        Dimensionality of face embeddings (128 for SFace).
    """

    def __init__(
        self,
        threshold: float = 0.45,
        embedding_dim: int = 128,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not installed.  Run: pip install faiss-cpu")

        self.threshold = threshold
        self.dim = embedding_dim

        # FAISS index – IndexFlatIP computes dot-products.
        # For L2-normalised vectors: dot-product == cosine similarity.
        self._index: Any = faiss.IndexFlatIP(self.dim)  # type: ignore[union-attr]

        # Parallel array: position i in the index → employee metadata dict
        self._id_map: List[Dict] = []

        self._lock = threading.RLock()

    # ── Index management ────────────────────────────────────

    def build_index(self, employees: List[Dict]) -> None:
        """
        (Re)build the FAISS index from a list of employee dicts.

        Each dict must have:
          employee_id, name, department, face_embedding (np.ndarray or bytes)
        """
        with self._lock:
            self._index = faiss.IndexFlatIP(self.dim)  # type: ignore[union-attr]
            self._id_map = []

            valid = []
            for emp in employees:
                emb = emp.get("face_embedding")
                if emb is None:
                    continue
                if isinstance(emb, bytes):
                    emb = np.frombuffer(emb, dtype=np.float32).copy()
                emb = emb.astype(np.float32)
                if emb.size != self.dim:
                    logger.warning(
                        "Skipping %s due to embedding dim mismatch (%d != %d)",
                        emp.get("employee_id", "?"),
                        emb.size,
                        self.dim,
                    )
                    continue
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                valid.append((emb, emp))

            if valid:
                embeddings_matrix = np.vstack([v[0] for v in valid])
                self._index.add(embeddings_matrix)
                self._id_map = [v[1] for v in valid]

        logger.info(
            "FAISS index built: %d employees indexed (threshold=%.2f)",
            len(self._id_map),
            self.threshold,
        )

    def add_employee(self, employee: Dict) -> None:
        """Add a single employee to an already-built index (no rebuild needed)."""
        emb = employee.get("face_embedding")
        if emb is None:
            logger.warning("Employee %s has no embedding – skipping.", employee.get("employee_id"))
            return

        if isinstance(emb, bytes):
            emb = np.frombuffer(emb, dtype=np.float32).copy()
        emb = emb.astype(np.float32)
        if emb.size != self.dim:
            logger.warning(
                "Employee %s embedding dim mismatch (%d != %d) – skipping.",
                employee.get("employee_id"),
                emb.size,
                self.dim,
            )
            return
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm

        with self._lock:
            self._index.add(emb.reshape(1, -1))
            self._id_map.append(employee)

        logger.debug("Employee added to index: %s", employee.get("employee_id"))

    def remove_employee(self, employee_id: str) -> None:
        """
        Remove an employee and rebuild the index.
        (FAISS IndexFlatIP does not support in-place deletion.)
        """
        with self._lock:
            new_map = [e for e in self._id_map if e.get("employee_id") != employee_id]
            if len(new_map) == len(self._id_map):
                logger.warning("Employee %s not found in index.", employee_id)
                return
            self.build_index(new_map)

    def clear(self) -> None:
        """Remove all entries from the index."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self.dim)  # type: ignore[union-attr]
            self._id_map = []

    # ── Recognition ─────────────────────────────────────────

    def recognize(
        self,
        embedding: np.ndarray,
        top_k: int = 1,
    ) -> RecognitionResult:
        """
        Match *embedding* against the indexed employees.

        Returns a RecognitionResult with is_known=True if the best match
        score ≥ self.threshold, otherwise is_known=False (UNKNOWN).
        """
        t0 = time.perf_counter()

        with self._lock:
            n_indexed = self._index.ntotal

        if n_indexed == 0:
            return RecognitionResult(
                employee_id=UNKNOWN_ID,
                name=UNKNOWN_LABEL,
                department="",
                confidence=0.0,
                is_known=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Ensure float32 and normalised
        query = embedding.astype(np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            return RecognitionResult(
                employee_id=UNKNOWN_ID,
                name=UNKNOWN_LABEL,
                department="",
                confidence=0.0,
                is_known=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        norm = np.linalg.norm(query)
        if norm > 0:
            query /= norm

        with self._lock:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query, k)

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        latency_ms = (time.perf_counter() - t0) * 1000

        if best_idx < 0 or best_score < self.threshold:
            return RecognitionResult(
                employee_id=UNKNOWN_ID,
                name=UNKNOWN_LABEL,
                department="",
                confidence=round(best_score, 4),
                is_known=False,
                latency_ms=round(latency_ms, 2),
            )

        with self._lock:
            emp = self._id_map[best_idx]

        return RecognitionResult(
            employee_id=emp.get("employee_id", UNKNOWN_ID),
            name=emp.get("name", UNKNOWN_LABEL),
            department=emp.get("department", ""),
            confidence=round(best_score, 4),
            is_known=True,
            latency_ms=round(latency_ms, 2),
        )

    def recognize_batch(
        self, embeddings: List[np.ndarray]
    ) -> List[RecognitionResult]:
        """Batch version of recognize()."""
        if not embeddings:
            return []

        t0 = time.perf_counter()

        with self._lock:
            n_indexed = self._index.ntotal

        if n_indexed == 0:
            return [
                RecognitionResult(UNKNOWN_ID, UNKNOWN_LABEL, "", 0.0, False)
                for _ in embeddings
            ]

        # Stack and normalise
        matrix = np.vstack([
            e.astype(np.float32) / (np.linalg.norm(e) + 1e-10)
            for e in embeddings
        ])

        with self._lock:
            scores, indices = self._index.search(matrix, 1)

        results: List[RecognitionResult] = []
        latency_each = (time.perf_counter() - t0) * 1000 / len(embeddings)

        for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            best_score = float(score_row[0])
            best_idx = int(idx_row[0])

            if best_idx < 0 or best_score < self.threshold:
                results.append(
                    RecognitionResult(
                        UNKNOWN_ID, UNKNOWN_LABEL, "",
                        round(best_score, 4), False,
                        round(latency_each, 2),
                    )
                )
            else:
                with self._lock:
                    emp = self._id_map[best_idx]
                results.append(
                    RecognitionResult(
                        emp.get("employee_id", UNKNOWN_ID),
                        emp.get("name", UNKNOWN_LABEL),
                        emp.get("department", ""),
                        round(best_score, 4),
                        True,
                        round(latency_each, 2),
                    )
                )

        return results

    # ── Info ────────────────────────────────────────────────

    @property
    def employee_count(self) -> int:
        with self._lock:
            return self._index.ntotal

    def get_all_employee_ids(self) -> List[str]:
        with self._lock:
            return [e.get("employee_id", "") for e in self._id_map]


# ─────────────────────────────────────────────────────────────
# Face Tracker  (debounce duplicate log entries)
# ─────────────────────────────────────────────────────────────

@dataclass
class _Track:
    employee_id: str
    last_seen: float
    center: Tuple[int, int]
    last_logged: float = field(default_factory=time.time)


class FaceTracker:
    """
    Simple centroid-based tracker that suppresses duplicate detection logs.

    An employee is only logged again if:
      • cooldown_seconds have elapsed since their last log, OR
      • they appear at a significantly different position (new encounter).
    """

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        max_distance: int = 80,
    ) -> None:
        self.cooldown = cooldown_seconds
        self.max_distance = max_distance
        self._tracks: Dict[str, _Track] = {}  # employee_id → track
        self._lock = threading.Lock()

    def should_log(
        self,
        result: RecognitionResult,
        center: Tuple[int, int],
    ) -> bool:
        """
        Returns True if this detection event should be written to the log.
        Updates internal track state regardless.

        Known employees  → tracked by employee_id (one track per person).
        Unknown persons  → tracked by position; each spatially distinct
                           unknown gets its own track so multiple unknowns
                           in the same frame are all logged independently.
        """
        now = time.time()

        if result.employee_id is not None:
            return self._should_log_known(result.employee_id, center, now)
        else:
            return self._should_log_unknown(center, now)

    def _should_log_known(self, emp_id: str, center: Tuple[int, int], now: float) -> bool:
        with self._lock:
            track = self._tracks.get(emp_id)
            if track is None:
                self._tracks[emp_id] = _Track(emp_id, now, center, now)
                return True
            track.last_seen = now
            track.center = center
            if now - track.last_logged >= self.cooldown:
                track.last_logged = now
                return True
        return False

    def _should_log_unknown(self, center: Tuple[int, int], now: float) -> bool:
        with self._lock:
            # Find the nearest existing unknown track within max_distance
            best_track = None
            best_dist = float(self.max_distance)
            for key, track in self._tracks.items():
                if not key.startswith("unk_"):
                    continue
                dist = math.hypot(center[0] - track.center[0], center[1] - track.center[1])
                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track is None:
                # New unknown at this position — create a unique track
                key = f"unk_{uuid.uuid4().hex[:8]}"
                self._tracks[key] = _Track(key, now, center, now)
                return True

            # Same unknown person seen before — apply cooldown
            best_track.last_seen = now
            best_track.center = center
            if now - best_track.last_logged >= self.cooldown:
                best_track.last_logged = now
                return True
        return False

    def cleanup_stale(self, max_age: float = 300.0) -> None:
        """Remove tracks that haven't been seen for *max_age* seconds."""
        now = time.time()
        with self._lock:
            stale = [eid for eid, t in self._tracks.items()
                     if now - t.last_seen > max_age]
            for eid in stale:
                del self._tracks[eid]

    @property
    def active_tracks(self) -> int:
        with self._lock:
            return len(self._tracks)
