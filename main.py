# main.py
import threading
import signal
from eye_module import track_face_and_move_eyes
from ai_voice_bot import voice_loop
from utils.servo_controller import kit

shutdown_event = threading.Event()

def signal_handler(sig, frame):
    print("\n[INFO] Shutting down...")
    shutdown_event.set()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    print("[INFO] System started.")

    eye_thread = threading.Thread(target=track_face_and_move_eyes, args=(shutdown_event,))
    eye_thread.start()

    try:
        voice_loop(shutdown_event)
    finally:
        shutdown_event.set()
        eye_thread.join()
        kit.servo[8].angle = None
        print("[INFO] System shutdown complete.")

if __name__ == "__main__":
    main()