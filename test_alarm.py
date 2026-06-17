"""
test_alarm.py  —  AlarmService quick test
Run:  python test_alarm.py
"""
import yaml
from modules.alarm_service import AlarmService

with open("config/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

alarm = AlarmService(cfg)
alarm.start()

print(f"AlarmService started")
print(f"  sound  : {alarm.sound}")
print(f"  output : {alarm.output}")
print(f"  text   : {alarm.voice_text}")
print(f"  cam    : {alarm.cam_host or '(not set)'}")
print()
print("Triggering alarm now...")

# Simulate an unknown-person detection event
alarm.on_detection_event({"is_known": False, "employee_name": "Unknown"})

# Wait for audio to finish (voice can take a few seconds)
import time
time.sleep(8)

alarm.stop()
print("Done.")
