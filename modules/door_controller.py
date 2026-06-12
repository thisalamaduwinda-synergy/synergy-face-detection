"""
door_controller.py
─────────────────────────────────────────────────────────────
Door unlock control triggered by successful face recognition.

Supported modes
───────────────
  gpio  — Direct Raspberry Pi GPIO pin (RPi.GPIO).
           Use when this software runs on the Pi itself.

  http  — HTTP GET/POST to a relay controller API.
           Use when the Pi (or any smart-relay) exposes an HTTP
           endpoint and this software runs on a separate machine.

  sdk   — Hikvision HCNetSDK NET_DVR_ControlGateway.
           Use when the camera itself controls an access-control relay.

Config keys (config.yaml  →  door: section)
───────────────────────────────────────────
  enabled          bool   – master switch (default false)
  mode             str    – "gpio", "http", or "sdk" (default "http")
  gpio_pin         int    – BCM GPIO pin number (default 18)
  open_duration    float  – seconds the relay stays energised (default 5)
  cooldown_seconds float  – min seconds between door triggers per employee
                            (default 10)
  min_confidence   float  – minimum match confidence to trigger (default 0.45)
  trigger_url      str    – HTTP mode: full URL to call (GET or POST)
  trigger_method   str    – "GET" or "POST" (default "GET")
  trigger_payload  dict   – optional JSON body for POST requests
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class DoorController:
    """
    Unlocks a door (via GPIO relay or HTTP request) when a known
    employee is recognised with sufficient confidence.

    Thread-safe.  Call trigger() from any recognition thread.
    """

    def __init__(self, cfg: Dict, sdk=None) -> None:
        door_cfg = cfg.get("door", {})

        self.enabled: bool          = bool(door_cfg.get("enabled", False))
        self.mode: str              = door_cfg.get("mode", "http").lower()
        self.gpio_pin: int          = int(door_cfg.get("gpio_pin", 18))
        self.open_duration: float   = float(door_cfg.get("open_duration", 5.0))
        self.cooldown: float        = float(door_cfg.get("cooldown_seconds", 10.0))
        self.min_confidence: float  = float(door_cfg.get("min_confidence", 0.45))
        self.trigger_url: str       = door_cfg.get("trigger_url", "").strip()
        self.trigger_method: str    = door_cfg.get("trigger_method", "GET").upper()
        self.trigger_payload: Dict  = door_cfg.get("trigger_payload", {})

        # HikvisionSDK instance (required when mode == "sdk")
        self._sdk = sdk

        # employee_id → timestamp of last successful trigger
        self._last_trigger: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Track current door state for the status API
        self._is_open: bool = False
        self._opened_by: Optional[str] = None
        self._opened_at: Optional[float] = None

        self._gpio_ok: bool = False

    # ── Lifecycle ────────────────────────────────────────────

    def initialize(self) -> None:
        """Set up GPIO if in gpio mode.  Call once at startup."""
        if not self.enabled:
            logger.info("DoorController disabled – skipping init")
            return

        if self.mode == "gpio":
            try:
                import RPi.GPIO as GPIO  # noqa: N813
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW)
                self._gpio_ok = True
                logger.info(
                    "DoorController (GPIO): pin BCM%d, open_duration=%.1fs",
                    self.gpio_pin, self.open_duration,
                )
            except ImportError:
                logger.error(
                    "RPi.GPIO not available – door mode=gpio requires the Pi. "
                    "Install with: pip install RPi.GPIO"
                )
            except Exception as exc:
                logger.error("GPIO init failed: %s", exc)

        elif self.mode == "http":
            if not self.trigger_url:
                logger.warning("DoorController (HTTP): trigger_url is empty")
            else:
                logger.info(
                    "DoorController (HTTP): %s %s, open_duration=%.1fs",
                    self.trigger_method, self.trigger_url, self.open_duration,
                )

        elif self.mode == "sdk":
            if self._sdk is None or not self._sdk.is_ready:
                logger.warning("DoorController (SDK): HikvisionSDK not ready")
            else:
                logger.info(
                    "DoorController (SDK): gateway=%d, open_duration=%.1fs",
                    self._sdk.gateway_index, self.open_duration,
                )

    def cleanup(self) -> None:
        """Release GPIO resources on shutdown."""
        if self.mode == "gpio" and self._gpio_ok:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup(self.gpio_pin)
                logger.info("GPIO pin BCM%d released", self.gpio_pin)
            except Exception:
                pass

    # ── Public trigger ───────────────────────────────────────

    def trigger(
        self,
        employee_id: str,
        employee_name: str,
        camera_id: str,
        confidence: float,
    ) -> bool:
        """
        Attempt to unlock the door.

        Returns True if the door was actually triggered this call,
        False if disabled / cooldown / confidence too low.
        """
        if not self.enabled:
            return False

        if confidence < self.min_confidence:
            return False

        now = time.time()
        with self._lock:
            last = self._last_trigger.get(employee_id, 0.0)
            if now - last < self.cooldown:
                return False
            self._last_trigger[employee_id] = now

        logger.info(
            "Door unlock triggered by %s (%s) from camera %s  conf=%.1f%%",
            employee_name, employee_id, camera_id, confidence * 100,
        )

        if self.mode == "gpio":
            threading.Thread(
                target=self._open_gpio,
                args=(employee_id, employee_name),
                daemon=True,
            ).start()
        elif self.mode == "sdk":
            threading.Thread(
                target=self._open_sdk,
                args=(employee_id, employee_name),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._open_http,
                args=(employee_id, employee_name),
                daemon=True,
            ).start()

        return True

    def manual_open(self) -> bool:
        """Manually unlock the door (from the API).  Ignores cooldown."""
        if not self.enabled:
            return False
        if self.mode == "gpio":
            threading.Thread(
                target=self._open_gpio, args=("MANUAL", "Manual"), daemon=True
            ).start()
        elif self.mode == "sdk":
            threading.Thread(
                target=self._open_sdk, args=("MANUAL", "Manual"), daemon=True
            ).start()
        else:
            threading.Thread(
                target=self._open_http, args=("MANUAL", "Manual"), daemon=True
            ).start()
        return True

    # ── Status ───────────────────────────────────────────────

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "mode": self.mode,
                "is_open": self._is_open,
                "opened_by": self._opened_by,
                "opened_at": self._opened_at,
                "open_duration": self.open_duration,
                "cooldown_seconds": self.cooldown,
                "min_confidence": self.min_confidence,
            }

    # ── GPIO implementation ──────────────────────────────────

    def _open_gpio(self, employee_id: str, employee_name: str) -> None:
        try:
            import RPi.GPIO as GPIO

            with self._lock:
                self._is_open = True
                self._opened_by = f"{employee_name} ({employee_id})"
                self._opened_at = time.time()

            GPIO.output(self.gpio_pin, GPIO.HIGH)
            logger.debug("GPIO BCM%d → HIGH (door open)", self.gpio_pin)

            time.sleep(self.open_duration)

            GPIO.output(self.gpio_pin, GPIO.LOW)
            logger.debug("GPIO BCM%d → LOW (door closed)", self.gpio_pin)

            with self._lock:
                self._is_open = False

        except Exception as exc:
            logger.error("GPIO door open failed: %s", exc)
            with self._lock:
                self._is_open = False

    # ── SDK implementation ───────────────────────────────────

    def _open_sdk(self, employee_id: str, employee_name: str) -> None:
        if self._sdk is None or not self._sdk.is_ready:
            logger.error("DoorController (SDK): SDK not ready")
            return
        try:
            with self._lock:
                self._is_open  = True
                self._opened_by = f"{employee_name} ({employee_id})"
                self._opened_at = time.time()

            ok = self._sdk.control_door(open=True)
            if not ok:
                logger.error("SDK door open failed for %s", employee_name)
                with self._lock:
                    self._is_open = False
                return

            time.sleep(self.open_duration)

            self._sdk.control_door(open=False)

            with self._lock:
                self._is_open = False

        except Exception as exc:
            logger.error("SDK door open error: %s", exc)
            with self._lock:
                self._is_open = False

    # ── HTTP implementation ──────────────────────────────────

    def _open_http(self, employee_id: str, employee_name: str) -> None:
        if not self.trigger_url:
            logger.warning("DoorController: trigger_url not configured")
            return
        try:
            with self._lock:
                self._is_open = True
                self._opened_by = f"{employee_name} ({employee_id})"
                self._opened_at = time.time()

            payload = dict(self.trigger_payload)
            payload.update({
                "employee_id": employee_id,
                "employee_name": employee_name,
                "action": "open",
            })

            with httpx.Client(timeout=5.0) as client:
                if self.trigger_method == "POST":
                    resp = client.post(self.trigger_url, json=payload)
                else:
                    resp = client.get(self.trigger_url, params=payload)

            logger.info(
                "Door HTTP trigger → %s %s  status=%d",
                self.trigger_method, self.trigger_url, resp.status_code,
            )

            # Auto-close after open_duration
            time.sleep(self.open_duration)

            with self._lock:
                self._is_open = False

        except Exception as exc:
            logger.error("HTTP door trigger failed: %s", exc)
            with self._lock:
                self._is_open = False
