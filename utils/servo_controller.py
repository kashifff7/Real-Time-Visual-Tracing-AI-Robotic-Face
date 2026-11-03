from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

def set_servo_angle(channel, angle):
    if 0 <= channel < 16:
        try:
            kit.servo[channel].angle = angle
        except Exception as e:
            print(f"[ERROR] Failed to set angle on channel {channel}: {e}")
    else:
        print(f"[ERROR] Invalid servo channel: {channel}")

def initialize_servos():
    print("[INFO] Initializing servos...")
    for ch in range(16):
        try:
            kit.servo[ch].angle = None
        except Exception:
            pass
    time.sleep(0.5)

def cleanup_servos():
    print("[INFO] Cleaning up servos...")
    for ch in range(16):
        try:
            kit.servo[ch].angle = None
        except Exception:
            pass
    time.sleep(0.5)
