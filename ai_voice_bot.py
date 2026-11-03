# ai_voice_bot.py
import os
import requests
import sounddevice as sd
import wave
import time
import threading
import pyttsx3
from phonemizer import phonemize
from utils.servo_controller import set_servo_angle, kit

# Constants
API_KEY = os.environ["OPENAI_API_KEY"]  # set on the Pi: export OPENAI_API_KEY=...
WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"

JAW_CHANNEL = 8
JAW_OPEN_ANGLE = 60
JAW_CLOSED_ANGLE = 115

# Initialize engine with slower speech
engine = pyttsx3.init()
engine.setProperty('rate', 120)  # Slower speech rate (default is 200)
engine.setProperty('volume', 0.9)

# Shared events for control
speaking_event = threading.Event()
listening_event = threading.Event()
stop_event = threading.Event()

def clean_text(text):
    """Clean text to remove unwanted characters"""
    if not text:
        return ""
    return ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()

def record_audio(duration=5, samplerate=16000):
    print("[INFO] Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    filename = "user_audio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return filename

def transcribe_audio(file_path):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {
        "file": (file_path, open(file_path, "rb")),
        "model": (None, "whisper-1"),
    }
    try:
        response = requests.post(WHISPER_URL, headers=headers, files=files, timeout=10)
        return clean_text(response.json().get("text", "").lower())
    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}")
        return ""

def chat_with_openai(prompt):
    if not prompt:
        return ""
    
    # Special commands
    if "stop" in prompt:
        return ""
    if "hello" in prompt:
        return "Hello commander, how can I help you today?"
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    json_data = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),  # gpt-3.5-turbo is retired; use OPENAI_MODEL=gpt-4o-mini
        "messages": [
            {"role": "system", "content": "You are a precise voice assistant. Respond concisely in 1-2 short sentences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(GPT_URL, headers=headers, json=json_data, timeout=15)
        return clean_text(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[ERROR] API request failed: {str(e)}")
        return "Sorry, I encountered an error."

def jaw_movement_loop():
    """Continuous jaw movement while speaking"""
    while speaking_event.is_set():
        try:
            set_servo_angle(JAW_CHANNEL, JAW_OPEN_ANGLE)
            time.sleep(0.5)
            if not speaking_event.is_set():
                break
            set_servo_angle(JAW_CHANNEL, JAW_CLOSED_ANGLE)
            time.sleep(0.5)
        except Exception as e:
            print(f"[JAW ERROR]: {str(e)}")
            break
    
    # Reset jaw position
    try:
        set_servo_angle(JAW_CHANNEL, JAW_CLOSED_ANGLE)
        time.sleep(0.1)
        kit.servo[JAW_CHANNEL].angle = None
    except:
        pass

def speak_with_jaw(text):
    """Robust speaking with continuous jaw movement"""
    if not text or stop_event.is_set():
        return
        
    text = clean_text(text)
    if not text:
        return
        
    print(f"[SPEAKING]: {text}")
    
    speaking_event.set()
    jaw_thread = threading.Thread(target=jaw_movement_loop)
    jaw_thread.start()
    
    try:
        engine.stop()  # Clear any pending speech
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[SPEECH ERROR]: {str(e)}")
    finally:
        speaking_event.clear()
        stop_event.clear()
        jaw_thread.join(timeout=2.0)
        try:
            set_servo_angle(JAW_CHANNEL, JAW_CLOSED_ANGLE)
            time.sleep(0.5)
            kit.servo[JAW_CHANNEL].angle = None
        except:
            pass

def voice_loop(shutdown_event):
    """Main voice processing loop"""
    speak_with_jaw("System ready. Awaiting your command.")
    
    while not shutdown_event.is_set():
        try:
            audio_path = record_audio()
            user_text = transcribe_audio(audio_path)
            
            if not user_text:
                continue
                
            print(f"[USER SAID]: {user_text}")
            
            if "sandy" in user_text:
                command = user_text.replace("sandy", "").strip()
                print(f"[PROCESSING COMMAND]: {command}")
                
                if "stop" in command:
                    stop_event.set()
                    speaking_event.clear()
                    engine.stop()
                    continue
                    
                if "hello" in command:
                    speak_with_jaw("Hello commander, how can I help you today?")
                    continue
                
                if command:
                    ai_response = chat_with_openai(command)
                    if ai_response:
                        print(f"[AI RESPONSE]: {ai_response}")
                        speak_with_jaw(ai_response)

        except Exception as e:
            print(f"[LOOP ERROR]: {str(e)}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        shutdown = threading.Event()
        voice_loop(shutdown)
    except KeyboardInterrupt:
        shutdown.set()
        speaking_event.clear()
        engine.stop()
        kit.servo[JAW_CHANNEL].angle = None
        print("\nSystem shut down gracefully")