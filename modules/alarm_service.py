"""
alarm_service.py
─────────────────────────────────────────────────────────────
Plays an alarm when an unknown person is detected by the
face recognition pipeline.

Sound modes
───────────
  beep   – Windows system beep (winsound.Beep)   → local only
  voice  – TTS spoken warning (pyttsx3)           → local / camera / both
  <path> – Path to a .wav file                    → local / camera / both

Output modes
────────────
  local   – PC / laptop speakers
  camera  – Hikvision IP camera built-in speaker (ISAPI Two-Way Audio)
  sdk     – Hikvision IP camera speaker via HCNetSDK Two-Way Audio
  both    – local AND camera simultaneously

Config keys (config.yaml  →  alarm: section)
────────────────────────────────────────────
  enabled           bool   – master switch (default false)
  cooldown_seconds  float  – min seconds between alarms (default 30)
  sound             str    – "beep", "voice", or path to WAV file
  output            str    – "local", "camera", or "both" (default "local")
  beep_frequency    int    – Hz for beep mode (default 1000)
  beep_duration     int    – ms for beep mode (default 1000)
  voice_text        str    – text spoken in voice / both mode
  camera_host       str    – Hikvision camera IP
  camera_user       str    – camera username (default "admin")
  camera_password   str    – camera password
  camera_channel    int    – Two-Way Audio channel (default 1)
"""

from __future__ import annotations

import audioop
import logging
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_SENTINEL = object()


class AlarmService:
    """
    Non-blocking alarm player for unknown-person detections.

    Subscribe on_detection_event to LoggingService so that every
    unknown-person event triggers the alarm with cooldown enforcement.
    """

    def __init__(self, cfg: Dict, sdk=None) -> None:
        alarm_cfg = cfg.get("alarm", {})
        greet_cfg = cfg.get("greeting", {})   # fallback for camera settings

        self.enabled: bool  = bool(alarm_cfg.get("enabled", False))
        self.cooldown: float = float(alarm_cfg.get("cooldown_seconds", 30.0))
        self.sound: str     = str(alarm_cfg.get("sound", "beep"))
        self.output: str    = str(alarm_cfg.get("output", "local")).lower()
        self.beep_frequency: int  = int(alarm_cfg.get("beep_frequency", 1000))
        self.beep_duration: int   = int(alarm_cfg.get("beep_duration", 1000))
        self.voice_text: str = str(
            alarm_cfg.get("voice_text", "Warning! Unknown person detected!")
        )

        # Camera settings — use alarm section, fall back to greeting section
        self.cam_host:     str = alarm_cfg.get("camera_host", greet_cfg.get("camera_host", ""))
        self.cam_user:     str = alarm_cfg.get("camera_user", greet_cfg.get("camera_user", "admin"))
        self.cam_password: str = alarm_cfg.get("camera_password", greet_cfg.get("camera_password", ""))
        self.cam_channel:  int = int(alarm_cfg.get("camera_channel", greet_cfg.get("camera_channel", 1)))

        # HikvisionSDK instance (used when output == "sdk")
        self._sdk = sdk

        self._last_alarm: float = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._worker: Optional[threading.Thread] = None

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            logger.info("AlarmService disabled")
            return
        self._worker = threading.Thread(
            target=self._alarm_loop,
            daemon=True,
            name="alarm-worker",
        )
        self._worker.start()
        logger.info(
            "AlarmService started (sound=%s, output=%s, cooldown=%.0fs)",
            self.sound, self.output, self.cooldown,
        )

    def stop(self) -> None:
        if self._worker and self._worker.is_alive():
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass
            self._worker.join(timeout=3)
        logger.info("AlarmService stopped")

    # ── Public API ───────────────────────────────────────────

    def on_detection_event(self, event: Dict) -> None:
        """Subscriber callback — called by LoggingService for every detection."""
        if not self.enabled or event.get("is_known"):
            return

        now = time.time()
        with self._lock:
            if now - self._last_alarm < self.cooldown:
                return
            self._last_alarm = now

        try:
            self._queue.put_nowait("alarm")
        except queue.Full:
            pass

    # ── Background worker ────────────────────────────────────

    def _alarm_loop(self) -> None:
        _com_init = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            _com_init = True
        except ImportError:
            pass

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            try:
                self._play_alarm()
            except Exception as exc:
                logger.warning("Alarm play error: %s", exc)
            finally:
                self._queue.task_done()

        if _com_init:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    # ── Dispatcher ───────────────────────────────────────────

    def _play_alarm(self) -> None:
        sound = self.sound.strip().lower()

        if sound == "beep":
            self._beep_alarm()
            return

        if sound == "voice":
            self._play_voice_alarm()
        else:
            self._play_wav_alarm(self.sound)

    # ── Beep ─────────────────────────────────────────────────

    def _beep_alarm(self) -> None:
        """Play beep locally and/or push synthesized tone to camera speaker."""
        # Local beep
        if self.output in ("local", "both"):
            try:
                import winsound
                winsound.Beep(self.beep_frequency, self.beep_duration)
                logger.info("Alarm: beep (%d Hz, %d ms)", self.beep_frequency, self.beep_duration)
            except Exception as exc:
                logger.warning("Alarm local beep failed: %s", exc)

        # Camera beep — synthesize sine wave → G.711 → ISAPI push
        if self.output in ("camera", "both", "sdk") and self.cam_host:
            cam_thread = threading.Thread(
                target=self._push_beep_to_camera,
                daemon=True,
            )
            cam_thread.start()
            cam_thread.join(timeout=10)

    def _push_beep_to_camera(self) -> None:
        """Generate a sine-wave beep as G.711 μ-law and push to camera speaker."""
        import math, struct
        try:
            sample_rate = 8000
            freq        = self.beep_frequency
            duration_s  = self.beep_duration / 1000.0
            num_samples = int(sample_rate * duration_s)

            # Generate 16-bit PCM sine wave
            pcm = struct.pack(
                f"<{num_samples}h",
                *[
                    int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
                    for i in range(num_samples)
                ],
            )

            # PCM 16-bit → G.711 μ-law
            ulaw = audioop.lin2ulaw(pcm, 2)

            self._push_wav_to_camera_ulaw(ulaw)
            logger.info("Alarm: beep pushed to camera (%d Hz, %d ms)", self.beep_frequency, self.beep_duration)

        except Exception as exc:
            logger.warning("Camera beep push failed: %s", exc)

    def _push_wav_to_camera_ulaw(self, ulaw_data: bytes) -> None:
        """Push raw G.711 μ-law bytes to Hikvision camera speaker via ISAPI."""
        base = f"http://{self.cam_host}"
        ch   = self.cam_channel
        auth = httpx.DigestAuth(self.cam_user, self.cam_password)

        xml_cfg = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<TwoWayAudioChannel version="2.0" '
            f'xmlns="http://www.hikvision.com/ver20/XMLSchema">'
            f'<id>{ch}</id><enabled>true</enabled>'
            f'<audioCompressionType>G.711ulaw</audioCompressionType>'
            f'<speakerVolume>100</speakerVolume><microphoneVolume>100</microphoneVolume>'
            f'<noisereduce>false</noisereduce>'
            f'<audioInputType>MicIn</audioInputType>'
            f'<audioOutputType>Speaker</audioOutputType>'
            f'</TwoWayAudioChannel>'
        )

        try:
            with httpx.Client(auth=auth) as client:
                client.put(
                    f"{base}/ISAPI/System/TwoWayAudio/channels/{ch}",
                    content=xml_cfg.encode(),
                    headers={"Content-Type": "application/xml"},
                    timeout=10,
                )
                try:
                    client.put(
                        f"{base}/ISAPI/System/TwoWayAudio/channels/{ch}/audioData",
                        content=ulaw_data,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=httpx.Timeout(connect=10, read=15, write=15, pool=10),
                    )
                except httpx.TimeoutException:
                    pass  # camera held connection open = audio accepted
        except Exception as exc:
            logger.warning("Camera beep ISAPI failed: %s", exc)

    # ── Voice alarm ──────────────────────────────────────────

    def _play_voice_alarm(self) -> None:
        try:
            import pyttsx3
        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            return

        try:
            engine = pyttsx3.init()
        except Exception as exc:
            logger.warning("pyttsx3 init failed: %s", exc)
            return

        if self.output == "sdk":
            self._speak_via_sdk(engine)
            return

        if self.output in ("camera", "both"):
            # Save TTS to temp WAV → push to camera in background thread
            tmp_wav = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_wav = f.name
                engine.save_to_file(self.voice_text, tmp_wav)
                engine.runAndWait()

                cam_thread = threading.Thread(
                    target=self._push_wav_to_camera,
                    args=(tmp_wav,),
                    daemon=True,
                )
                cam_thread.start()

                if self.output == "both":
                    # Play locally at the same time the camera thread runs
                    self._local_voice(engine)

                cam_thread.join(timeout=15)

            except Exception as exc:
                logger.warning("Voice alarm (camera) failed: %s", exc)
            finally:
                if tmp_wav and os.path.exists(tmp_wav):
                    try:
                        os.unlink(tmp_wav)
                    except Exception:
                        pass
        else:
            # local only
            self._local_voice(engine)

    def _local_voice(self, engine) -> None:
        try:
            engine.say(self.voice_text)
            engine.runAndWait()
            logger.info("Alarm: voice played on local speakers")
        except Exception as exc:
            logger.warning("Local voice alarm failed: %s", exc)

    # ── WAV file alarm ───────────────────────────────────────

    def _play_wav_alarm(self, path: str) -> None:
        if self.output in ("camera", "both"):
            cam_thread = threading.Thread(
                target=self._push_wav_to_camera,
                args=(path,),
                daemon=True,
            )
            cam_thread.start()

            if self.output == "both":
                self._local_wav(path)

            cam_thread.join(timeout=15)
        else:
            self._local_wav(path)

    def _local_wav(self, path: str) -> None:
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            logger.info("Alarm: WAV played on local speakers (%s)", path)
        except Exception as exc:
            logger.warning("Local WAV alarm failed (%s): %s", path, exc)

    # ── SDK audio push ───────────────────────────────────────

    def _speak_via_sdk(self, engine) -> None:
        """Generate TTS WAV then push PCM to camera via HCNetSDK Two-Way Audio."""
        if self._sdk is None or not self._sdk.is_ready:
            logger.warning("AlarmService (SDK): SDK not ready — falling back to local")
            self._local_voice(engine)
            return

        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            engine.save_to_file(self.voice_text, tmp_wav)
            engine.runAndWait()

            pcm = self._wav_to_pcm8k(tmp_wav)
            if not pcm:
                return

            if not self._sdk.start_audio():
                logger.warning("AlarmService (SDK): audio session failed")
                return

            chunk = 320  # 20 ms at 8 kHz 16-bit mono
            for i in range(0, len(pcm), chunk):
                self._sdk.send_audio(pcm[i:i + chunk])

            self._sdk.stop_audio()
            logger.info("Alarm: SDK audio sent")

        except Exception as exc:
            logger.warning("AlarmService SDK speak error: %s", exc)
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass

    def _wav_to_pcm8k(self, wav_path: str):
        """Read WAV and return 8 kHz 16-bit mono PCM bytes."""
        try:
            with wave.open(wav_path, "rb") as wf:
                n_ch = wf.getnchannels()
                sw   = wf.getsampwidth()
                rate = wf.getframerate()
                pcm  = wf.readframes(wf.getnframes())
            if sw == 1:
                pcm = audioop.lin2lin(pcm, 1, 2); sw = 2
            elif sw == 4:
                pcm = audioop.lin2lin(pcm, 4, 2); sw = 2
            if n_ch == 2:
                pcm = audioop.tomono(pcm, sw, 0.5, 0.5)
            if rate != 8000:
                pcm, _ = audioop.ratecv(pcm, sw, 1, rate, 8000, None)
            return pcm
        except Exception as exc:
            logger.error("WAV→PCM conversion error: %s", exc)
            return None

    # ── Hikvision camera audio push ──────────────────────────

    def _push_wav_to_camera(self, wav_path: str) -> None:
        """Convert WAV → G.711 μ-law and push to Hikvision camera speaker."""
        if not self.cam_host:
            logger.warning("alarm.camera_host not set — skipping camera audio")
            return

        ulaw_data = self._wav_to_ulaw(wav_path)
        if not ulaw_data:
            return

        base = f"http://{self.cam_host}"
        ch   = self.cam_channel
        auth = httpx.DigestAuth(self.cam_user, self.cam_password)

        xml_cfg = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<TwoWayAudioChannel version="2.0" '
            f'xmlns="http://www.hikvision.com/ver20/XMLSchema">'
            f'<id>{ch}</id><enabled>true</enabled>'
            f'<audioCompressionType>G.711ulaw</audioCompressionType>'
            f'<speakerVolume>100</speakerVolume><microphoneVolume>100</microphoneVolume>'
            f'<noisereduce>false</noisereduce>'
            f'<audioInputType>MicIn</audioInputType>'
            f'<audioOutputType>Speaker</audioOutputType>'
            f'</TwoWayAudioChannel>'
        )

        try:
            with httpx.Client(auth=auth) as client:
                client.put(
                    f"{base}/ISAPI/System/TwoWayAudio/channels/{ch}",
                    content=xml_cfg.encode(),
                    headers={"Content-Type": "application/xml"},
                    timeout=10,
                )
                try:
                    r = client.put(
                        f"{base}/ISAPI/System/TwoWayAudio/channels/{ch}/audioData",
                        content=ulaw_data,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=httpx.Timeout(connect=10, read=20, write=20, pool=10),
                    )
                    logger.info(
                        "Alarm: camera audio pushed (%d bytes → %s, HTTP %d)",
                        len(ulaw_data), self.cam_host, r.status_code,
                    )
                except httpx.TimeoutException:
                    logger.info(
                        "Alarm: camera audio streamed (%d bytes → %s, connection held = OK)",
                        len(ulaw_data), self.cam_host,
                    )
        except Exception as exc:
            logger.warning("Camera audio push failed: %s", exc)

    def _wav_to_ulaw(self, wav_path: str) -> Optional[bytes]:
        """Convert a WAV file to G.711 μ-law mono 8 kHz bytes."""
        try:
            with wave.open(wav_path, "rb") as wf:
                n_ch = wf.getnchannels()
                sw   = wf.getsampwidth()
                rate = wf.getframerate()
                pcm  = wf.readframes(wf.getnframes())

            if sw == 1:
                pcm = audioop.lin2lin(pcm, 1, 2); sw = 2
            elif sw == 4:
                pcm = audioop.lin2lin(pcm, 4, 2); sw = 2

            if n_ch == 2:
                pcm = audioop.tomono(pcm, sw, 0.5, 0.5)

            if rate != 8000:
                pcm, _ = audioop.ratecv(pcm, sw, 1, rate, 8000, None)

            return audioop.lin2ulaw(pcm, 2)

        except Exception as exc:
            logger.error("WAV→G.711 conversion error: %s", exc)
            return None
