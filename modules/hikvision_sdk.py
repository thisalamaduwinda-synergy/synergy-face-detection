"""
hikvision_sdk.py
─────────────────────────────────────────────────────────────
Python ctypes wrapper for Hikvision HCNetSDK.dll

Covers
──────
  • SDK init / login / logout
  • Real-time video frame capture  (H.264 stream callback)
  • Two-Way Audio output           (voice / alarm → camera speaker)
  • Alarm event callbacks          (motion, intrusion, line-crossing …)
  • Door / relay control           (NET_DVR_ControlGateway)

SDK download
────────────
  https://www.hikvision.com/en/support/download/sdk/
  Place HCNetSDK.dll and all companion DLLs in one of:
    • <project>/lib/hikvision/
    • config.yaml  →  hikvision_sdk.sdk_path

Config keys (config.yaml  →  hikvision_sdk:)
────────────────────────────────────────────
  enabled     bool – master switch (default false)
  sdk_path    str  – folder containing HCNetSDK.dll
  host        str  – camera IP
  port        int  – camera port (default 8000)
  username    str  – camera username
  password    str  – camera password
  channel     int  – video / audio channel number (default 1)
  gateway_index int – door/relay index for ControlGateway (default 1)
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

NET_DVR_DEV_ADDRESS_MAX_LEN    = 129
NET_DVR_LOGIN_USERNAME_MAX_LEN = 64
NET_DVR_LOGIN_PASSWD_MAX_LEN   = 64
SERIALNO_LEN                   = 48

# RealData callback data types
NET_DVR_SYSHEAD       = 1
NET_DVR_STREAMDATA    = 2
NET_DVR_AUDIOSTREAMDATA = 3

# Door control states
GATEWAY_CLOSE = 0
GATEWAY_OPEN  = 1

# Alarm command codes (NET_DVR_SetDVRMessageCallBack)
COMM_ALARM_V30               = 0x1880
COMM_ALARM_RULE              = 0x1102
COMM_UPLOAD_FACESNAP_RESULT  = 0x2702
COMM_GATEKEEPER_NOTIFY       = 0x6801
COMM_MOTION_DETECTION        = 0x3000

# Audio PCM format sent to SDK
AUDIO_SAMPLE_RATE  = 8000
AUDIO_SAMPLE_WIDTH = 2        # 16-bit
AUDIO_CHANNELS     = 1

# ─────────────────────────────────────────────────────────────
# ctypes callback types  (stdcall / WINAPI on Windows)
# ─────────────────────────────────────────────────────────────

if sys.platform == "win32":
    _FUNCTYPE = ctypes.WINFUNCTYPE
else:
    _FUNCTYPE = ctypes.CFUNCTYPE

REALDATACALLBACK_V30 = _FUNCTYPE(
    None,
    ctypes.c_long,                       # lRealHandle
    ctypes.c_uint32,                     # dwDataType
    ctypes.POINTER(ctypes.c_ubyte),      # pBuffer
    ctypes.c_uint32,                     # dwBufSize
    ctypes.c_void_p,                     # pUser
)

MSGCALLBACK = _FUNCTYPE(
    None,
    ctypes.c_long,    # lCommand
    ctypes.c_void_p,  # pAlarmer  (NET_DVR_ALARMER*)
    ctypes.c_char_p,  # pAlarmInfo
    ctypes.c_uint32,  # dwBufLen
    ctypes.c_void_p,  # pUser
)

VOICECOMCALLBACK = _FUNCTYPE(
    None,
    ctypes.c_long,    # lVoiceComHandle
    ctypes.c_char_p,  # pRecvDataBuffer
    ctypes.c_uint32,  # dwBufSize
    ctypes.c_ubyte,   # byAudioFlag  0=recv 1=send
    ctypes.c_void_p,  # pUser
)

# ─────────────────────────────────────────────────────────────
# Structures
# ─────────────────────────────────────────────────────────────

class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber",       ctypes.c_ubyte * SERIALNO_LEN),
        ("byAlarmInPortNum",    ctypes.c_ubyte),
        ("byAlarmOutPortNum",   ctypes.c_ubyte),
        ("byDiskNum",           ctypes.c_ubyte),
        ("byDVRType",           ctypes.c_ubyte),
        ("byChanNum",           ctypes.c_ubyte),
        ("byStartChan",         ctypes.c_ubyte),
        ("byAudioChanNum",      ctypes.c_ubyte),
        ("byIPChanNum",         ctypes.c_ubyte),
        ("byZeroChanNum",       ctypes.c_ubyte),
        ("byMainProto",         ctypes.c_ubyte),
        ("bySubProto",          ctypes.c_ubyte),
        ("bySupport",           ctypes.c_ubyte),
        ("bySupport1",          ctypes.c_ubyte),
        ("bySupport2",          ctypes.c_ubyte),
        ("wDevType",            ctypes.c_uint16),
        ("bySupport3",          ctypes.c_ubyte),
        ("byMultiStreamProto",  ctypes.c_ubyte),
        ("byStartDChan",        ctypes.c_ubyte),
        ("byStartDTalkChan",    ctypes.c_ubyte),
        ("byHighDChanNum",      ctypes.c_ubyte),
        ("bySupport4",          ctypes.c_ubyte),
        ("byLanguageType",      ctypes.c_ubyte),
        ("byVoiceInChanNum",    ctypes.c_ubyte),
        ("byStartVoiceInChanNo",ctypes.c_ubyte),
        ("byRes3",              ctypes.c_ubyte * 2),
        ("byMirrorChanNum",     ctypes.c_ubyte),
        ("wStartMirrorChanNo",  ctypes.c_uint16),
        ("byRes2",              ctypes.c_ubyte),
    ]


class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
    _fields_ = [
        ("struDeviceV30",      NET_DVR_DEVICEINFO_V30),
        ("bySupportLock",      ctypes.c_ubyte),
        ("byRetryLoginTime",   ctypes.c_ubyte),
        ("byPasswordLevel",    ctypes.c_ubyte),
        ("byProxyType",        ctypes.c_ubyte),
        ("dwSurplusLockTime",  ctypes.c_uint32),
        ("byCharEncodeType",   ctypes.c_ubyte),
        ("bySupportDev5",      ctypes.c_ubyte),
        ("bySupport",          ctypes.c_ubyte),
        ("byLoginMode",        ctypes.c_ubyte),
        ("dwOEMCode",          ctypes.c_uint32),
        ("iVoiceInChanNum",    ctypes.c_int),
        ("iStartVoiceInChanNo",ctypes.c_int),
        ("byRes2",             ctypes.c_ubyte * 4),
        ("byMirrorChanNum",    ctypes.c_ubyte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("byRes3",             ctypes.c_ubyte * 2),
    ]


class NET_DVR_USER_LOGIN_INFO(ctypes.Structure):
    _fields_ = [
        ("sDeviceAddress", ctypes.c_char * NET_DVR_DEV_ADDRESS_MAX_LEN),
        ("byUseTransport", ctypes.c_ubyte),
        ("wPort",          ctypes.c_uint16),
        ("sUserName",      ctypes.c_char * NET_DVR_LOGIN_USERNAME_MAX_LEN),
        ("sPassword",      ctypes.c_char * NET_DVR_LOGIN_PASSWD_MAX_LEN),
        ("cbLoginResult",  ctypes.c_void_p),   # NULL = synchronous login
        ("pUser",          ctypes.c_void_p),
        ("bUseAsynLogin",  ctypes.c_int),
        ("byProxyType",    ctypes.c_ubyte),
        ("byVerifyMode",   ctypes.c_ubyte),
        ("byLoginMode",    ctypes.c_ubyte),
        ("byHttps",        ctypes.c_ubyte),
        ("iProxyID",       ctypes.c_long),
        ("byVerifyKey",    ctypes.c_ubyte * 32),
        ("byRes",          ctypes.c_ubyte * 119),
    ]


class NET_DVR_PREVIEWINFO(ctypes.Structure):
    _fields_ = [
        ("lChannel",           ctypes.c_long),
        ("dwStreamType",       ctypes.c_uint32),  # 0=main 1=sub 2=tri
        ("dwLinkMode",         ctypes.c_uint32),  # 0=TCP 1=UDP
        ("hPlayWnd",           ctypes.c_void_p),  # NULL = no display
        ("bBlocked",           ctypes.c_uint32),
        ("bPassbackRecord",    ctypes.c_uint32),
        ("byPreviewMode",      ctypes.c_ubyte),
        ("byStreamID",         ctypes.c_ubyte * 32),
        ("byProtoType",        ctypes.c_ubyte),   # 0=private 1=RTSP
        ("byRes1",             ctypes.c_ubyte),
        ("byVideoCodingType",  ctypes.c_ubyte),
        ("dwDisplayBufNum",    ctypes.c_uint32),
        ("byNPQMode",          ctypes.c_ubyte),
        ("byRes",              ctypes.c_ubyte * 215),
    ]


class NET_DVR_SETUPALARM_PARAM_V41(ctypes.Structure):
    _fields_ = [
        ("dwSize",             ctypes.c_uint32),
        ("byLevel",            ctypes.c_ubyte),   # 0=one-level 1=two-level 2=three
        ("byAlarmInfoType",    ctypes.c_ubyte),   # 0=old 1=V40
        ("byRetAlarmTypeV40",  ctypes.c_ubyte),
        ("byRes1",             ctypes.c_ubyte),
        ("byDeployType",       ctypes.c_ubyte),   # 0=software 1=hardware
        ("byRes",              ctypes.c_ubyte * 251),
    ]


# ─────────────────────────────────────────────────────────────
# Main SDK wrapper
# ─────────────────────────────────────────────────────────────

class HikvisionSDK:
    """
    Thread-safe wrapper around Hikvision HCNetSDK.dll.

    Usage
    ─────
    sdk = HikvisionSDK(cfg)
    if sdk.load() and sdk.initialize() and sdk.login():
        sdk.start_alarm(on_event)
        sdk.start_audio(channel=1)
        sdk.control_door(open=True)
        ...
        sdk.cleanup()
    """

    def __init__(self, cfg: Dict) -> None:
        sdk_cfg = cfg.get("hikvision_sdk", {})
        greet_cfg = cfg.get("greeting", {})    # fallback for camera creds

        self.enabled:       bool  = bool(sdk_cfg.get("enabled", False))
        self.sdk_path:      str   = sdk_cfg.get("sdk_path", "lib/hikvision")
        self.host:          str   = sdk_cfg.get("host", greet_cfg.get("camera_host", ""))
        self.port:          int   = int(sdk_cfg.get("port", 8000))
        self.username:      str   = sdk_cfg.get("username", greet_cfg.get("camera_user", "admin"))
        self.password:      str   = sdk_cfg.get("password", greet_cfg.get("camera_password", ""))
        self.channel:       int   = int(sdk_cfg.get("channel", 1))
        self.gateway_index: int   = int(sdk_cfg.get("gateway_index", 1))

        self._dll:          Optional[Any] = None
        self._user_id:      int   = -1
        self._lock               = threading.Lock()
        self._real_handles: Dict[int, int] = {}
        self._alarm_handle: int   = -1
        self._voice_handle: int   = -1

        # Hold references to prevent garbage collection (ctypes callbacks must not be GC'd)
        self._cb_frame: Optional[Any] = None
        self._cb_alarm: Optional[Any] = None
        self._cb_voice: Optional[Any] = None

        self._ready:    bool  = False

    # ── Properties ───────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready and self._user_id >= 0

    @property
    def _d(self) -> Any:
        """Narrowed, non-None DLL handle. Only call after load()."""
        assert self._dll is not None, "HCNetSDK DLL not loaded"
        return self._dll

    # ── DLL loading ──────────────────────────────────────────

    def load(self) -> bool:
        """Search for and load HCNetSDK.dll."""
        if not self.enabled:
            logger.info("HikvisionSDK disabled")
            return False
        if not sys.platform.startswith("win"):
            logger.error("HikvisionSDK is Windows-only")
            return False

        search = [
            Path(self.sdk_path),
            Path("lib") / "hikvision",
            Path("C:/Program Files/Hikvision/HCNetSDK"),
            Path("C:/HCNetSDK"),
            Path(os.environ.get("HIKVISION_SDK_PATH", "")),
        ]
        for folder in search:
            dll_file = folder / "HCNetSDK.dll"
            if not dll_file.is_file():
                continue
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(folder.resolve()))
                self._dll = ctypes.WinDLL(str(dll_file.resolve()))
                self._setup_prototypes()
                logger.info("HCNetSDK loaded: %s", dll_file)
                return True
            except OSError as exc:
                logger.warning("SDK load failed (%s): %s", dll_file, exc)

        logger.error(
            "HCNetSDK.dll not found. Download the SDK from "
            "https://www.hikvision.com/en/support/download/sdk/ "
            "and place the DLLs in %s", self.sdk_path
        )
        return False

    def _setup_prototypes(self) -> None:
        """Declare return / arg types for SDK functions."""
        d: Any = self._d

        d.NET_DVR_Init.restype               = ctypes.c_bool
        d.NET_DVR_Cleanup.restype            = ctypes.c_bool
        d.NET_DVR_SetConnectTime.restype     = ctypes.c_bool
        d.NET_DVR_SetReconnect.restype       = ctypes.c_bool
        d.NET_DVR_GetLastError.restype       = ctypes.c_uint32

        d.NET_DVR_Login_V40.restype          = ctypes.c_long
        d.NET_DVR_Login_V40.argtypes         = [
            ctypes.POINTER(NET_DVR_USER_LOGIN_INFO),
            ctypes.POINTER(NET_DVR_DEVICEINFO_V40),
        ]
        d.NET_DVR_Logout.restype             = ctypes.c_bool
        d.NET_DVR_Logout.argtypes            = [ctypes.c_long]

        d.NET_DVR_RealPlay_V40.restype       = ctypes.c_long
        d.NET_DVR_RealPlay_V40.argtypes      = [
            ctypes.c_long,
            ctypes.POINTER(NET_DVR_PREVIEWINFO),
            REALDATACALLBACK_V30,
            ctypes.c_void_p,
        ]
        d.NET_DVR_StopRealPlay.restype       = ctypes.c_bool
        d.NET_DVR_StopRealPlay.argtypes      = [ctypes.c_long]

        d.NET_DVR_SetDVRMessageCallBack_V31.restype  = ctypes.c_bool
        d.NET_DVR_SetDVRMessageCallBack_V31.argtypes = [MSGCALLBACK, ctypes.c_void_p]
        d.NET_DVR_SetupAlarmChan_V41.restype = ctypes.c_long
        d.NET_DVR_SetupAlarmChan_V41.argtypes= [
            ctypes.c_long,
            ctypes.POINTER(NET_DVR_SETUPALARM_PARAM_V41),
        ]
        d.NET_DVR_CloseAlarmChan_V30.restype = ctypes.c_bool
        d.NET_DVR_CloseAlarmChan_V30.argtypes= [ctypes.c_long]

        d.NET_DVR_StartVoiceCom_MR_V30.restype  = ctypes.c_long
        d.NET_DVR_StartVoiceCom_MR_V30.argtypes = [
            ctypes.c_long,   # lUserID
            ctypes.c_uint32, # dwVoiceChan
            VOICECOMCALLBACK,
            ctypes.c_void_p,
        ]
        d.NET_DVR_VoiceComSendData.restype   = ctypes.c_bool
        d.NET_DVR_VoiceComSendData.argtypes  = [
            ctypes.c_long,   # lVoiceComHandle
            ctypes.c_char_p, # pSendBuf
            ctypes.c_uint32, # dwBufSize
        ]
        d.NET_DVR_StopVoiceCom.restype       = ctypes.c_bool
        d.NET_DVR_StopVoiceCom.argtypes      = [ctypes.c_long]

        d.NET_DVR_ControlGateway.restype     = ctypes.c_bool
        d.NET_DVR_ControlGateway.argtypes    = [
            ctypes.c_long,   # lUserID
            ctypes.c_long,   # lGatewayIndex
            ctypes.c_uint32, # dwStaic  0=close 1=open
        ]

    # ── Lifecycle ────────────────────────────────────────────

    def initialize(self) -> bool:
        """Call NET_DVR_Init and configure timeouts."""
        if not self._dll:
            return False
        if not self._d.NET_DVR_Init():
            logger.error("NET_DVR_Init failed")
            return False
        self._d.NET_DVR_SetConnectTime(2000, 1)
        self._d.NET_DVR_SetReconnect(10000, ctypes.c_bool(True))
        logger.info("HCNetSDK initialized")
        return True

    def login(self) -> bool:
        """Login to the configured camera device."""
        if not self._dll or not self.host:
            return False

        login_info = NET_DVR_USER_LOGIN_INFO()
        ctypes.memset(ctypes.addressof(login_info), 0, ctypes.sizeof(login_info))
        login_info.sDeviceAddress = self.host.encode()
        login_info.wPort          = self.port
        login_info.sUserName      = self.username.encode()
        login_info.sPassword      = self.password.encode()
        login_info.bUseAsynLogin  = 0   # synchronous

        device_info = NET_DVR_DEVICEINFO_V40()
        ctypes.memset(ctypes.addressof(device_info), 0, ctypes.sizeof(device_info))

        uid = self._d.NET_DVR_Login_V40(
            ctypes.byref(login_info),
            ctypes.byref(device_info),
        )
        if uid < 0:
            logger.error(
                "SDK login failed: host=%s:%d  error=%d",
                self.host, self.port,
                self._d.NET_DVR_GetLastError(),
            )
            return False

        with self._lock:
            self._user_id = uid
            self._ready   = True
        logger.info("SDK login OK: host=%s:%d  user_id=%d", self.host, self.port, uid)
        return True

    def logout(self) -> None:
        with self._lock:
            if self._user_id >= 0 and self._dll:
                self._d.NET_DVR_Logout(self._user_id)
            self._user_id = -1
            self._ready   = False

    def cleanup(self) -> None:
        """Stop all sessions and release the SDK."""
        self.stop_alarm()
        self.stop_audio()
        for ch in list(self._real_handles.keys()):
            self.stop_preview(ch)
        self.logout()
        if self._dll:
            self._d.NET_DVR_Cleanup()
        logger.info("HikvisionSDK cleanup done")

    # ── Video capture ────────────────────────────────────────

    def start_preview(
        self,
        on_data: Callable[[bytes, int], None],
        channel: Optional[int] = None,
        stream_type: int = 1,
    ) -> bool:
        """
        Start real-time video capture.

        on_data(raw_bytes, data_type) is called for each SDK packet:
          data_type == NET_DVR_SYSHEAD    → H.264 SPS/PPS header
          data_type == NET_DVR_STREAMDATA → H.264 frame data
        """
        if not self.is_ready:
            return False
        ch = channel or self.channel

        preview = NET_DVR_PREVIEWINFO()
        ctypes.memset(ctypes.addressof(preview), 0, ctypes.sizeof(preview))
        preview.lChannel      = ch
        preview.dwStreamType  = stream_type  # 1 = sub-stream
        preview.dwLinkMode    = 0            # TCP
        preview.hPlayWnd      = None
        preview.bBlocked      = 1
        preview.byProtoType   = 0            # private protocol

        def _cb(handle, data_type, buf, buf_size, user):
            if buf_size > 0 and buf:
                try:
                    on_data(bytes(buf[:buf_size]), data_type)
                except Exception as exc:
                    logger.debug("Preview callback error: %s", exc)

        cb = REALDATACALLBACK_V30(_cb)
        self._cb_frame = cb   # prevent GC

        handle = self._d.NET_DVR_RealPlay_V40(
            self._user_id,
            ctypes.byref(preview),
            cb,
            None,
        )
        if handle < 0:
            logger.error(
                "SDK preview start failed: ch=%d  error=%d",
                ch, self._d.NET_DVR_GetLastError(),
            )
            return False

        self._real_handles[ch] = handle
        logger.info("SDK preview started: ch=%d  handle=%d", ch, handle)
        return True

    def stop_preview(self, channel: Optional[int] = None) -> None:
        ch = channel or self.channel
        handle = self._real_handles.pop(ch, -1)
        if handle >= 0 and self._dll:
            self._d.NET_DVR_StopRealPlay(handle)

    # ── Alarm / event callbacks ──────────────────────────────

    def start_alarm(self, on_event: Callable[[Dict], None]) -> bool:
        """
        Register an alarm callback for motion, intrusion, line-crossing, etc.

        on_event receives a dict:
          {
            "command": int,       # COMM_* constant
            "command_name": str,  # human-readable command name
            "raw": bytes,         # raw alarm info bytes
          }
        """
        if not self.is_ready:
            return False

        def _cb(command, alarmer, alarm_info, buf_len, user):
            try:
                raw = bytes(alarm_info[:buf_len]) if alarm_info and buf_len > 0 else b""
                on_event({
                    "command":      command,
                    "command_name": _ALARM_NAMES.get(command, f"0x{command:04X}"),
                    "raw":          raw,
                })
            except Exception as exc:
                logger.debug("Alarm callback error: %s", exc)

        cb = MSGCALLBACK(_cb)
        self._cb_alarm = cb

        self._d.NET_DVR_SetDVRMessageCallBack_V31(cb, None)

        param = NET_DVR_SETUPALARM_PARAM_V41()
        ctypes.memset(ctypes.addressof(param), 0, ctypes.sizeof(param))
        param.dwSize          = ctypes.sizeof(param)
        param.byLevel         = 1    # two-level alarm
        param.byAlarmInfoType = 1    # V40 alarm info
        param.byDeployType    = 0    # software arming

        handle = self._d.NET_DVR_SetupAlarmChan_V41(
            self._user_id,
            ctypes.byref(param),
        )
        if handle < 0:
            logger.error(
                "SDK alarm setup failed: error=%d",
                self._d.NET_DVR_GetLastError(),
            )
            return False

        self._alarm_handle = handle
        logger.info("SDK alarm channel armed: handle=%d", handle)
        return True

    def stop_alarm(self) -> None:
        if self._alarm_handle >= 0 and self._dll:
            self._d.NET_DVR_CloseAlarmChan_V30(self._alarm_handle)
            self._alarm_handle = -1

    # ── Two-Way Audio ────────────────────────────────────────

    def start_audio(
        self,
        channel: Optional[int] = None,
        on_recv: Optional[Callable[[bytes], None]] = None,
    ) -> bool:
        """
        Open a two-way audio session to the camera speaker.
        Call send_audio() afterwards to push PCM data.
        on_recv(pcm_bytes) is called when mic audio arrives (optional).
        """
        if not self.is_ready:
            return False
        ch = channel or self.channel

        def _cb(handle, buf, buf_size, flag, user):
            if flag == 0 and on_recv and buf_size > 0 and buf:
                try:
                    on_recv(bytes(buf[:buf_size]))
                except Exception as exc:
                    logger.debug("Audio recv callback error: %s", exc)

        cb = VOICECOMCALLBACK(_cb)
        self._cb_voice = cb

        handle = self._d.NET_DVR_StartVoiceCom_MR_V30(
            self._user_id,
            ch,
            cb,
            None,
        )
        if handle < 0:
            logger.error(
                "SDK audio start failed: ch=%d  error=%d",
                ch, self._d.NET_DVR_GetLastError(),
            )
            return False

        self._voice_handle = handle
        logger.info("SDK audio started: ch=%d  handle=%d", ch, handle)
        return True

    def send_audio(self, pcm_data: bytes) -> bool:
        """
        Send raw PCM audio to the camera speaker.
        Format: 8 kHz, 16-bit, mono (matches AUDIO_SAMPLE_RATE/WIDTH/CHANNELS).
        """
        if self._voice_handle < 0 or not self._dll:
            return False
        buf = ctypes.create_string_buffer(pcm_data)
        ok = self._d.NET_DVR_VoiceComSendData(
            self._voice_handle,
            buf,
            len(pcm_data),
        )
        if not ok:
            logger.warning(
                "NET_DVR_VoiceComSendData failed: error=%d",
                self._d.NET_DVR_GetLastError(),
            )
        return bool(ok)

    def stop_audio(self) -> None:
        if self._voice_handle >= 0 and self._dll:
            self._d.NET_DVR_StopVoiceCom(self._voice_handle)
            self._voice_handle = -1

    # ── Door control ─────────────────────────────────────────

    def control_door(self, open: bool = True, gateway_index: Optional[int] = None) -> bool:
        """Open or close a door/relay via NET_DVR_ControlGateway."""
        if not self.is_ready:
            logger.warning("SDK not ready — door control skipped")
            return False
        idx   = gateway_index if gateway_index is not None else self.gateway_index
        state = GATEWAY_OPEN if open else GATEWAY_CLOSE
        ok    = self._d.NET_DVR_ControlGateway(self._user_id, idx, state)
        if not ok:
            logger.error(
                "SDK door control failed: gw=%d  error=%d",
                idx, self._d.NET_DVR_GetLastError(),
            )
            return False
        logger.info("SDK door %s: gateway=%d", "opened" if open else "closed", idx)
        return True


# ─────────────────────────────────────────────────────────────
# Alarm command name lookup
# ─────────────────────────────────────────────────────────────

_ALARM_NAMES: Dict[int, str] = {
    COMM_ALARM_V30:              "alarm_v30",
    COMM_ALARM_RULE:             "alarm_rule",
    COMM_UPLOAD_FACESNAP_RESULT: "face_snap",
    COMM_GATEKEEPER_NOTIFY:      "gatekeeper",
    COMM_MOTION_DETECTION:       "motion_detection",
    0x2800: "plate_result",
    0x3704: "people_counting",
    0x1900: "shelter_alarm",
    0x2100: "disk_full",
    0x2200: "disk_error",
}
