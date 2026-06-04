"""
Face Recognition System – modules package
"""

from modules.camera_stream import CameraStream, MultiCameraManager, Frame
from modules.face_detector import FaceDetector, DetectedFace
from modules.face_embedding import FaceEmbedding
from modules.face_recognizer import FaceRecognizer, RecognitionResult
from modules.employee_database import EmployeeDatabase
from modules.logging_service import LoggingService

__all__ = [
    "CameraStream",
    "MultiCameraManager",
    "Frame",
    "FaceDetector",
    "DetectedFace",
    "FaceEmbedding",
    "FaceRecognizer",
    "RecognitionResult",
    "EmployeeDatabase",
    "LoggingService",
]
