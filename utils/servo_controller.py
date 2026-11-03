#servo_controller.py
from adafruit_servokit import ServoKit
import time

# Initialize the ServoKit instance with 16 channels (PCA9685)
kit = ServoKit(channels=16)

def set_servo_angle(channel, angle):
    """
    Safely sets the angle of a given servo channel.
    If angle is None, disables the servo (no signal).
    """
    if 0 <= channel < 16:
        try:
            kit.servo[channel].angle = angle
        except Exception as e:
            print(f"[ERROR] Failed to set angle on channel {channel}: {e}")
    else:
        print(f"[ERROR] Invalid servo channel: {channel}")

def initialize_servos():
    """
    Initialize all servo channels to a safe off state (disabled).
    """
    print("[INFO] Initializing servos...")
    for ch in range(16):
        try:
            kit.servo[ch].angle = None
        except Exception:
            pass
    time.sleep(0.5)

def cleanup_servos():
    """
    Safely disable all servos during system shutdown.
    """
    print("[INFO] Cleaning up servos...")
    for ch in range(16):
        try:
            kit.servo[ch].angle = None
        except Exception:
            pass
    time.sleep(0.5)