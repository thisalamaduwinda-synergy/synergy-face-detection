"""
tests/test_recognition.py
─────────────────────────────────────────────────────────────
Unit tests for core recognition modules.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────
# FaceEmbedding tests
# ─────────────────────────────────────────────────────────────

class TestFaceEmbedding:
    def setup_method(self):
        from modules.face_embedding import FaceEmbedding
        self.FaceEmbedding = FaceEmbedding

    def test_embedding_to_bytes_roundtrip(self):
        fe = self.FaceEmbedding.__new__(self.FaceEmbedding)
        original = np.random.randn(512).astype(np.float32)
        raw = self.FaceEmbedding.embedding_to_bytes(original)
        restored = self.FaceEmbedding.bytes_to_embedding(raw, dim=512)
        np.testing.assert_array_almost_equal(original, restored, decimal=5)

    def test_bytes_to_embedding_wrong_dim(self):
        fe = self.FaceEmbedding.__new__(self.FaceEmbedding)
        raw = np.zeros(256, dtype=np.float32).tobytes()
        with pytest.raises(ValueError, match="Expected 512-d"):
            self.FaceEmbedding.bytes_to_embedding(raw, dim=512)

    def test_cosine_similarity_identical(self):
        v = np.random.randn(512).astype(np.float32)
        sim = self.FaceEmbedding.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = self.FaceEmbedding.cosine_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_cosine_similarity_range(self):
        for _ in range(20):
            a = np.random.randn(512).astype(np.float32)
            b = np.random.randn(512).astype(np.float32)
            sim = self.FaceEmbedding.cosine_similarity(a, b)
            assert -1.01 <= sim <= 1.01

    def test_normalise(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        n = self.FaceEmbedding._normalise(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_normalise_zero_vector(self):
        v = np.zeros(512, dtype=np.float32)
        n = self.FaceEmbedding._normalise(v)
        # Should not raise; norm remains 0
        assert np.all(n == 0)


# ─────────────────────────────────────────────────────────────
# FaceRecognizer tests
# ─────────────────────────────────────────────────────────────

class TestFaceRecognizer:
    def setup_method(self):
        from modules.face_recognizer import FaceRecognizer, UNKNOWN_ID
        self.FaceRecognizer = FaceRecognizer
        self.UNKNOWN_ID = UNKNOWN_ID

    def _make_recognizer(self, threshold=0.45):
        return self.FaceRecognizer(threshold=threshold, embedding_dim=512)

    def _random_unit(self):
        v = np.random.randn(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_empty_index_returns_unknown(self):
        rec = self._make_recognizer()
        emb = self._random_unit()
        result = rec.recognize(emb)
        assert not result.is_known
        assert result.employee_id == self.UNKNOWN_ID

    def test_single_employee_recognised(self):
        rec = self._make_recognizer(threshold=0.3)
        emb = self._random_unit()
        employees = [{
            "employee_id": "EMP001",
            "name": "Alice",
            "department": "Engineering",
            "face_embedding": emb.copy(),
        }]
        rec.build_index(employees)
        result = rec.recognize(emb)
        assert result.is_known
        assert result.employee_id == "EMP001"
        assert result.name == "Alice"
        assert result.confidence > 0.99

    def test_different_embedding_returns_unknown(self):
        rec = self._make_recognizer(threshold=0.99)
        emb1 = self._random_unit()
        emb2 = self._random_unit()
        # Force orthogonal vectors
        emb2 = emb2 - np.dot(emb2, emb1) * emb1
        emb2 /= np.linalg.norm(emb2)
        rec.build_index([{
            "employee_id": "EMP001",
            "name": "Alice",
            "department": "",
            "face_embedding": emb1,
        }])
        result = rec.recognize(emb2)
        assert not result.is_known

    def test_add_employee_increments_count(self):
        rec = self._make_recognizer()
        assert rec.employee_count == 0
        rec.add_employee({
            "employee_id": "EMP001",
            "name": "Alice",
            "department": "",
            "face_embedding": self._random_unit(),
        })
        assert rec.employee_count == 1

    def test_remove_employee(self):
        rec = self._make_recognizer()
        emb = self._random_unit()
        rec.build_index([{
            "employee_id": "EMP001",
            "name": "Alice",
            "department": "",
            "face_embedding": emb,
        }])
        assert rec.employee_count == 1
        rec.remove_employee("EMP001")
        assert rec.employee_count == 0

    def test_batch_recognize(self):
        rec = self._make_recognizer(threshold=0.3)
        emb = self._random_unit()
        rec.build_index([{
            "employee_id": "EMP001",
            "name": "Alice",
            "department": "",
            "face_embedding": emb,
        }])
        results = rec.recognize_batch([emb, self._random_unit()])
        assert len(results) == 2
        assert results[0].is_known

    def test_clear_index(self):
        rec = self._make_recognizer()
        rec.add_employee({
            "employee_id": "E1", "name": "X", "department": "",
            "face_embedding": self._random_unit(),
        })
        rec.clear()
        assert rec.employee_count == 0

    def test_bytes_embedding_accepted(self):
        rec = self._make_recognizer(threshold=0.3)
        emb = self._random_unit()
        rec.build_index([{
            "employee_id": "E1",
            "name": "Bob",
            "department": "HR",
            "face_embedding": emb.tobytes(),
        }])
        result = rec.recognize(emb)
        assert result.is_known


# ─────────────────────────────────────────────────────────────
# FaceTracker tests
# ─────────────────────────────────────────────────────────────

class TestFaceTracker:
    def setup_method(self):
        from modules.face_recognizer import FaceTracker, RecognitionResult
        self.FaceTracker = FaceTracker
        self.RecognitionResult = RecognitionResult

    def _result(self, is_known=True, employee_id="E1"):
        return self.RecognitionResult(
            employee_id=employee_id,
            name="Alice",
            department="Eng",
            confidence=0.9,
            is_known=is_known,
        )

    def test_first_detection_always_logged(self):
        tracker = self.FaceTracker(cooldown_seconds=30)
        assert tracker.should_log(self._result(), (100, 100))

    def test_second_detection_within_cooldown_not_logged(self):
        tracker = self.FaceTracker(cooldown_seconds=30)
        tracker.should_log(self._result(), (100, 100))
        assert not tracker.should_log(self._result(), (100, 100))

    def test_unknown_persons_tracked_separately_per_camera(self):
        from modules.face_recognizer import UNKNOWN_ID
        tracker = self.FaceTracker(cooldown_seconds=30)
        r1 = self._result(is_known=False, employee_id=UNKNOWN_ID)
        assert tracker.should_log(r1, (50, 50))
        # Second unknown from a different position is still the same key
        # so should NOT be logged again within cooldown
        assert not tracker.should_log(r1, (55, 55))


# ─────────────────────────────────────────────────────────────
# EmployeeDatabase tests
# ─────────────────────────────────────────────────────────────

class TestEmployeeDatabase:
    def setup_method(self):
        from modules.employee_database import EmployeeDatabase
        self._tmp = tempfile.mkdtemp()
        self.db = EmployeeDatabase(f"sqlite:///{self._tmp}/test.db")
        self.db.initialize()

    def test_add_and_retrieve_employee(self):
        self.db.add_employee("E001", "Alice", "Engineering")
        emp = self.db.get_employee("E001")
        assert emp is not None
        assert emp["name"] == "Alice"
        assert emp["department"] == "Engineering"

    def test_duplicate_employee_raises(self):
        self.db.add_employee("E002", "Bob", "HR")
        with pytest.raises(ValueError):
            self.db.add_employee("E002", "Bob2", "Finance")

    def test_update_embedding(self):
        self.db.add_employee("E003", "Carol", "IT")
        emb = np.random.randn(512).astype(np.float32)
        ok = self.db.update_employee_embedding("E003", emb)
        assert ok
        emps = self.db.get_all_employees_with_embeddings()
        found = next(e for e in emps if e["employee_id"] == "E003")
        np.testing.assert_array_almost_equal(
            found["face_embedding"] / np.linalg.norm(found["face_embedding"]),
            emb / np.linalg.norm(emb),
            decimal=4,
        )

    def test_deactivate_hides_employee(self):
        self.db.add_employee("E004", "Dave", "Sales")
        self.db.deactivate_employee("E004")
        emp = self.db.get_employee("E004")
        assert emp is None

    def test_employee_count(self):
        for i in range(5):
            self.db.add_employee(f"EMP{i:03d}", f"Person {i}", "Dept")
        assert self.db.employee_count() == 5

    def test_log_detection(self):
        entry = self.db.log_detection(
            camera_id="cam_001",
            employee_id="E001",
            employee_name="Alice",
            confidence=0.82,
            is_known=True,
            bbox=[10, 20, 100, 150],
        )
        assert entry.id is not None
        logs = self.db.get_recent_logs(limit=5)
        assert len(logs) >= 1
        assert logs[0]["camera_id"] == "cam_001"

    def test_get_stats(self):
        self.db.add_employee("S001", "Stats Person", "IT")
        self.db.log_detection("cam_001", "S001", "Stats Person", 0.9, True)
        self.db.log_detection("cam_001", None, "Unknown", 0.2, False)
        stats = self.db.get_detection_stats()
        assert stats["total_detections"] == 2
        assert stats["known_detections"] == 1
        assert stats["unknown_detections"] == 1
