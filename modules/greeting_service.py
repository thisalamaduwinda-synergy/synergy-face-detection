"""
greeting_service.py
─────────────────────────────────────────────────────────────
Speaks a personalized greeting ("Hi, <name>") when a known
employee is recognised by the face recognition pipeline.

Output modes
────────────
  local   – plays through PC speakers (edge-tts neural voice or pyttsx3)
  camera  – pushes G.711 audio to a Hikvision IP camera's
             built-in speaker via ISAPI Two-Way Audio API

Config keys (config.yaml  →  greeting: section)
───────────────────────────────────────────────
  enabled           bool   – master switch (default false)
  template          str    – greeting text, {name} = employee name
  cooldown_seconds  float  – min seconds between greetings per employee
  output            str    – "local" or "camera" (default "local")
  tts_engine        str    – "edge" (neural, requires internet) or "pyttsx3"
  voice_name        str    – edge-tts voice, e.g. "en-US-AriaNeural"
  voice_rate        int    – pyttsx3 speed in wpm  (default 160)
  voice_volume      float  – volume 0.0–1.0        (default 0.9)
  voice_index       int    – pyttsx3 voice index   (default 0)

  # Camera output only:
  camera_host       str    – Hikvision camera IP
  camera_user       str    – camera username (default "admin")
  camera_password   str    – camera password
  camera_channel    int    – two-way audio channel (default 1)
"""

from __future__ import annotations

import asyncio
import audioop
import logging
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Dict, List, Optional, cast

import httpx

logger = logging.getLogger(__name__)

_SENTINEL = object()


class GreetingService:
    """
    Non-blocking text-to-speech greeter.

    Call greet(employee_id, employee_name) from any thread;
    audio plays asynchronously without blocking recognition.
    """

    def __init__(self, cfg: Dict, sdk=None) -> None:
        greet_cfg = cfg.get("greeting", {})

        self.enabled:         bool  = bool(greet_cfg.get("enabled", False))
        self.template:        str   = greet_cfg.get("template", "Hi, {name}")
        self.cooldown:        float = float(greet_cfg.get("cooldown_seconds", 60.0))
        self.output:          str   = greet_cfg.get("output", "local").lower()
        self.tts_engine:      str   = greet_cfg.get("tts_engine", "pyttsx3").lower()
        self.voice_name:      str   = greet_cfg.get("voice_name", "en-US-AriaNeural")
        self.voice_rate:      int   = int(greet_cfg.get("voice_rate", 160))
        self.voice_volume:    float = float(greet_cfg.get("voice_volume", 0.9))
        self.voice_index:     int   = int(greet_cfg.get("voice_index", 0))

        # Camera output settings (ISAPI path)
        self.cam_host:     str = greet_cfg.get("camera_host", "")
        self.cam_user:     str = greet_cfg.get("camera_user", "admin")
        self.cam_password: str = greet_cfg.get("camera_password", "")
        self.cam_channel:  int = int(greet_cfg.get("camera_channel", 1))

        # HikvisionSDK instance (used when output == "sdk")
        self._sdk = sdk

        self._last_greeted: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._worker: Optional[threading.Thread] = None

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            logger.info("GreetingService disabled")
            return
        self._worker = threading.Thread(
            target=self._audio_loop,
            daemon=True,
            name="greeting-tts",
        )
        self._worker.start()
        logger.info(
            "GreetingService started (output=%s, cooldown=%.0fs)",
            self.output, self.cooldown,
        )

    def stop(self) -> None:
        if self._worker and self._worker.is_alive():
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass
            self._worker.join(timeout=3)
        logger.info("GreetingService stopped")

    # ── Public API ───────────────────────────────────────────

    def greet(self, employee_id: str, employee_name: str) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        with self._cooldown_lock:
            last = self._last_greeted.get(employee_id, 0.0)
            if now - last < self.cooldown:
                return False
            self._last_greeted[employee_id] = now

        text = self.template.format(name=employee_name)
        try:
            self._queue.put_nowait(text)
            logger.debug("Greeting queued: %s", text)
            return True
        except queue.Full:
            logger.debug("Greeting queue full – skipped for %s", employee_name)
            return False

    # ── Background audio loop ────────────────────────────────

    def _audio_loop(self) -> None:
        if self.output == "local" and self.tts_engine == "edge":
            self._audio_loop_edge()
        else:
            self._audio_loop_pyttsx3()

    def _audio_loop_edge(self) -> None:
        """Audio loop using edge-tts neural voices + pygame playback."""
        try:
            import pygame
            pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=1024)
            pygame.mixer.init()
            logger.info("GreetingService edge-tts ready (voice=%s)", self.voice_name)
        except Exception as exc:
            logger.error("pygame init failed: %s — falling back to pyttsx3", exc)
            self._audio_loop_pyttsx3()
            return

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            try:
                self._speak_edge(item)
                logger.debug("edge-tts spoke: %s", item)
            except Exception as exc:
                logger.warning("edge-tts speak error: %s", exc)
            finally:
                self._queue.task_done()

        try:
            pygame.mixer.quit()
        except Exception:
            pass

    def _speak_edge(self, text: str) -> None:
        """Generate speech with edge-tts and play via pygame."""
        import pygame
        import edge_tts

        tmp_mp3 = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_mp3 = f.name

            async def _generate() -> None:
                communicate = edge_tts.Communicate(text, voice=self.voice_name)
                await communicate.save(tmp_mp3)

            asyncio.run(_generate())

            pygame.mixer.music.load(tmp_mp3)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        finally:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                try:
                    pygame.mixer.music.unload()
                    os.unlink(tmp_mp3)
                except Exception:
                    pass

    def _audio_loop_pyttsx3(self) -> None:
        """Audio loop using pyttsx3 / Windows SAPI."""
        # Windows COM init — required for pyttsx3 in non-main threads
        _com_init = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            _com_init = True
        except ImportError:
            pass

        try:
            import pyttsx3
        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            return

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.voice_rate)
            engine.setProperty("volume", self.voice_volume)
            voices: List = cast(List, engine.getProperty("voices") or [])
            if voices and self.voice_index < len(voices):
                engine.setProperty("voice", voices[self.voice_index].id)
                logger.info("TTS voice: %s", voices[self.voice_index].name)
            logger.info("GreetingService pyttsx3 ready (output=%s)", self.output)
        except Exception as exc:
            logger.error("pyttsx3 engine init failed: %s", exc)
            return

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                break

            try:
                if self.output == "sdk":
                    self._speak_via_sdk(engine, item)
                elif self.output == "camera":
                    self._speak_to_camera(engine, item, pyttsx3)
                else:
                    # Re-initialise the engine for every utterance.
                    # Windows SAPI silently stops responding after the first
                    # runAndWait() when running in a non-main thread; a fresh
                    # engine instance per call is the reliable workaround.
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    engine = pyttsx3.init()
                    engine.setProperty("rate",   self.voice_rate)
                    engine.setProperty("volume", self.voice_volume)
                    voices: List = cast(List, engine.getProperty("voices") or [])
                    if voices and self.voice_index < len(voices):
                        engine.setProperty("voice", voices[self.voice_index].id)
                    engine.say(item)
                    engine.runAndWait()
                    logger.debug("TTS spoke: %s", item)
            except Exception as exc:
                logger.warning("TTS speak error: %s", exc)
            finally:
                self._queue.task_done()

        if _com_init:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    # ── SDK audio push ───────────────────────────────────────

    def _speak_via_sdk(self, engine, text: str) -> None:
        """Generate TTS WAV then push PCM to camera via HCNetSDK Two-Way Audio."""
        if self._sdk is None or not self._sdk.is_ready:
            logger.warning("GreetingService (SDK): SDK not ready — falling back to local")
            engine.say(text)
            engine.runAndWait()
            return

        import os, tempfile, wave, audioop
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name
            engine.save_to_file(text, tmp_wav)
            engine.runAndWait()

            pcm = self._wav_to_pcm8k(tmp_wav)
            if not pcm:
                return

            if not self._sdk.start_audio(channel=self.cam_channel):
                logger.warning("GreetingService (SDK): audio session failed")
                return

            # Send in 160-byte chunks (20 ms at 8 kHz 16-bit mono)
            chunk = 320
            for i in range(0, len(pcm), chunk):
                self._sdk.send_audio(pcm[i:i + chunk])

            self._sdk.stop_audio()
            logger.info("GreetingService: SDK audio sent for '%s'", text)

        except Exception as exc:
            logger.warning("GreetingService SDK speak error: %s", exc)
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass

    def _wav_to_pcm8k(self, wav_path: str) -> Optional[bytes]:
        """Read WAV and return 8 kHz 16-bit mono PCM bytes."""
        import wave, audioop
        try:
            with wave.open(wav_path, "rb") as wf:
                n_ch = wf.getnchannels()
                sw   = wf.getsampwidth()
                rate = wf.getframerate()
                pcm: bytes = wf.readframes(wf.getnframes())
            if sw == 1:
                pcm = bytes(audioop.lin2lin(pcm, 1, 2)); sw = 2
            elif sw == 4:
                pcm = bytes(audioop.lin2lin(pcm, 4, 2)); sw = 2
            if n_ch == 2:
                pcm = bytes(audioop.tomono(pcm, sw, 0.5, 0.5))
            if rate != 8000:
                pcm = bytes(audioop.ratecv(pcm, sw, 1, rate, 8000, None)[0])
            return pcm
        except Exception as exc:
            logger.error("WAV→PCM conversion error: %s", exc)
            return None

    # ── Camera audio push (Hikvision ISAPI) ──────────────────

    def _speak_to_camera(self, engine, text: str, pyttsx3_mod) -> None:
        """
        1. Save TTS to a temp WAV file
        2. Convert PCM → G.711 μ-law (8 kHz mono)
        3. Push to Hikvision camera via ISAPI Two-Way Audio
        """
        if not self.cam_host:
            logger.warning("camera_host not set — falling back to local speaker")
            engine.say(text)
            engine.runAndWait()
            return

        tmp_wav = None
        try:
            # ── Step 1: Generate WAV via pyttsx3 ─────────────
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav = f.name

            engine.save_to_file(text, tmp_wav)
            engine.runAndWait()

            # ── Step 2: Read WAV and convert to G.711 μ-law ──
            ulaw_data = self._wav_to_ulaw(tmp_wav)
            if not ulaw_data:
                logger.error("WAV conversion failed — no audio data")
                return

            # ── Step 3: Push to Hikvision camera ─────────────
            self._hikvision_push(ulaw_data)

        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass

    def _wav_to_ulaw(self, wav_path: str) -> Optional[bytes]:
        """Convert a WAV file to G.711 μ-law mono 8kHz bytes."""
        try:
            with wave.open(wav_path, "rb") as wf:
                n_ch    = wf.getnchannels()
                sw      = wf.getsampwidth()   # bytes per sample
                rate    = wf.getframerate()
                pcm     = wf.readframes(wf.getnframes())

            # Ensure 16-bit samples
            if sw == 1:
                pcm = bytes(audioop.lin2lin(pcm, 1, 2))
                sw = 2
            elif sw == 4:
                pcm = bytes(audioop.lin2lin(pcm, 4, 2))
                sw = 2

            # Stereo → mono
            if n_ch == 2:
                pcm = bytes(audioop.tomono(pcm, sw, 0.5, 0.5))

            # Resample to 8000 Hz
            if rate != 8000:
                pcm = bytes(audioop.ratecv(pcm, sw, 1, rate, 8000, None)[0])

            # PCM 16-bit → G.711 μ-law 8-bit
            ulaw: bytes = bytes(audioop.lin2ulaw(pcm, 2))
            return ulaw

        except Exception as exc:
            logger.error("WAV→G.711 conversion error: %s", exc)
            return None

    def _hikvision_push(self, ulaw_data: bytes) -> None:
        """Push G.711 audio to Hikvision camera speaker via ISAPI Two-Way Audio."""
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
            f'<audioInputType>MicIn</audioInputType><audioOutputType>Speaker</audioOutputType>'
            f'</TwoWayAudioChannel>'
        )

        with httpx.Client(auth=auth) as client:
            # Configure channel
            client.put(
                f"{base}/ISAPI/System/TwoWayAudio/channels/{ch}",
                content=xml_cfg.encode(),
                headers={"Content-Type": "application/xml"},
                timeout=10,
            )

            # Push audio directly — more reliable than session-based approach
            try:
                r = client.put(
                    f"{base}/ISAPI/System/TwoWayAudio/channels/{ch}/audioData",
                    content=ulaw_data,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=httpx.Timeout(connect=10, read=20, write=20, pool=10),
                )
                logger.info(
                    "Hikvision audio pushed: %d bytes → %s (HTTP %d)",
                    len(ulaw_data), self.cam_host, r.status_code,
                )
            except httpx.TimeoutException:
                # Camera held connection open — streaming accepted, audio played
                logger.info(
                    "Hikvision audio streamed: %d bytes → %s (connection held = OK)",
                    len(ulaw_data), self.cam_host,
                )
