# eye_module.py
import cv2
import time
from picamera2 import Picamera2
from utils.servo_controller import set_servo_angle

# Servo channels
HORIZONTAL_EYE = 0
VERTICAL_EYE = 1
LEFT_UPPER_LID = 2
RIGHT_UPPER_LID = 3
LEFT_LOWER_LID = 4
RIGHT_LOWER_LID = 5

def map_range(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def move_eyeballs(x, y, width, height):
    hor = map_range(x, 0, width, 60, 120)  # Horizontal movement remains same
    ver = map_range(y, 0, height, 120, 60)  # Inverted vertical mapping (120 when y=0, 60 when y=height)
    set_servo_angle(HORIZONTAL_EYE, hor)
    set_servo_angle(VERTICAL_EYE, ver)

def blink():
    set_servo_angle(LEFT_UPPER_LID, 10)
    set_servo_angle(RIGHT_UPPER_LID, 10)
    set_servo_angle(LEFT_LOWER_LID, 50)
    set_servo_angle(RIGHT_LOWER_LID, 50)
    time.sleep(0.2)
    set_servo_angle(LEFT_UPPER_LID, 50)
    set_servo_angle(RIGHT_UPPER_LID, 50)
    set_servo_angle(LEFT_LOWER_LID, 10)
    set_servo_angle(RIGHT_LOWER_LID, 10)

def track_face_and_move_eyes(shutdown_event):
    try:
        cam = Picamera2()
        config = cam.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
        cam.configure(config)
        cam.start()
    except Exception as e:
        print(f"[ERROR] Camera failed to initialize: {e}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    blink_timer = time.time()

    try:
        while not shutdown_event.is_set():
            frame = cam.capture_array()
            frame = cv2.flip(frame, 1)  # Invert horizontally for natural motion
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cx, cy = x + w // 2, y + h // 2
                move_eyeballs(cx, cy, 320, 240)
                
                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw crosshair at center point
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                
            elif time.time() - blink_timer > 10:
                blink()
                blink_timer = time.time()

            # Show the frame with face detection
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Eye module terminated.")