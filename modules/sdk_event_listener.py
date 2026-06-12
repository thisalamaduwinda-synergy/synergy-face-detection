"""
sdk_event_listener.py
─────────────────────────────────────────────────────────────
Listens to Hikvision SDK alarm callbacks and dispatches
structured events to the rest of the system.

Events dispatched
─────────────────
  motion_detected   – camera motion-detection triggered
  intrusion         – perimeter / virtual fence breach
  line_crossing     – virtual line crossed
  face_snap         – camera uploaded a face snapshot
  gatekeeper        – access-control face match result

Each listener callback receives a dict:
  {
    "event_type": str,          # one of the names above
    "command":    int,          # raw SDK command code
    "camera_id":  str,          # config camera id
    "timestamp":  float,        # time.time()
    "raw":        bytes,        # raw alarm info bytes
  }
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Callable, Dict, List, Optional

from modules.hikvision_sdk import (
    HikvisionSDK,
    COMM_ALARM_V30,
    COMM_ALARM_RULE,
    COMM_UPLOAD_FACESNAP_RESULT,
    COMM_GATEKEEPER_NOTIFY,
    COMM_MOTION_DETECTION,
)

logger = logging.getLogger(__name__)

# ── Alarm type codes inside NET_DVR_ALARMINFO_V30 ────────────
# (byte offset 4 in the raw alarm info payload)
_ALARM_TYPE_MOTION    = 0
_ALARM_TYPE_IO        = 1
_ALARM_TYPE_SHELTER   = 5
_ALARM_TYPE_INTRUSION = 14   # VCA perimeter
_ALARM_TYPE_LINE      = 15   # VCA line crossing

# Map SDK command → event_type
_CMD_TO_EVENT: Dict[int, str] = {
    COMM_ALARM_V30:              "alarm_v30",
    COMM_ALARM_RULE:             "vca_rule",
    COMM_UPLOAD_FACESNAP_RESULT: "face_snap",
    COMM_GATEKEEPER_NOTIFY:      "gatekeeper",
    COMM_MOTION_DETECTION:       "motion_detected",
}


class SDKEventListener:
    """
    Wraps HikvisionSDK.start_alarm() and translates raw SDK
    alarm payloads into typed event dicts for subscribers.
    """

    def __init__(self, sdk: HikvisionSDK, camera_id: str = "sdk_cam") -> None:
        self._sdk       = sdk
        self._camera_id = camera_id
        self._lock      = threading.Lock()
        self._subs: List[Callable[[Dict], None]] = []

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> bool:
        if not self._sdk.is_ready:
            logger.warning("SDKEventListener: SDK not ready — skipping alarm setup")
            return False
        ok = self._sdk.start_alarm(self._on_sdk_event)
        if ok:
            logger.info("SDKEventListener started for camera '%s'", self._camera_id)
        return ok

    def stop(self) -> None:
        self._sdk.stop_alarm()
        logger.info("SDKEventListener stopped")

    # ── Subscription ─────────────────────────────────────────

    def subscribe(self, callback: Callable[[Dict], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    def unsubscribe(self, callback: Callable[[Dict], None]) -> None:
        with self._lock:
            try:
                self._subs.remove(callback)
            except ValueError:
                pass

    # ── Internal ─────────────────────────────────────────────

    def _on_sdk_event(self, raw_event: Dict) -> None:
        command = raw_event.get("command", 0)
        raw     = raw_event.get("raw", b"")

        event_type = _CMD_TO_EVENT.get(command, f"unknown_0x{command:04X}")

        # Refine COMM_ALARM_V30 using the alarm type byte in the payload
        if command == COMM_ALARM_V30 and len(raw) >= 5:
            alarm_type = raw[4]
            if alarm_type == _ALARM_TYPE_MOTION:
                event_type = "motion_detected"
            elif alarm_type == _ALARM_TYPE_INTRUSION:
                event_type = "intrusion"
            elif alarm_type == _ALARM_TYPE_LINE:
                event_type = "line_crossing"
            elif alarm_type == _ALARM_TYPE_SHELTER:
                event_type = "shelter_alarm"

        event = {
            "event_type": event_type,
            "command":    command,
            "camera_id":  self._camera_id,
            "timestamp":  time.time(),
            "raw":        raw,
        }

        logger.info(
            "SDK event: type=%s  camera=%s  command=0x%04X",
            event_type, self._camera_id, command,
        )

        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception as exc:
                logger.debug("SDKEventListener subscriber error: %s", exc)
