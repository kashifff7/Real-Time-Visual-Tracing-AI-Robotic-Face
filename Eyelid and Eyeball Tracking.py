from eyelid_and_tracking import run_eyelid_and_tracking from jaw_movement import speak_with_jaw
from ai_voice_exchange import record_audio, transcribe_audio, chat_with_openai
import os import threading

def main():
print("Voice Assistant is ready. Press Ctrl+C to stop.")

tracking_thread = threading.Thread(target=run_eyelid_and_tracking) tracking_thread.start()

speak_with_jaw("Hello! How can I assist you today?")

while True:
user_audio_path = record_audio(duration=10)

user_input = transcribe_audio(user_audio_path) print(f"You said: {user_input}")

if user_input.strip().lower() == "exit": speak_with_jaw("Goodbye!") os.remove(user_audio_path)
break

ai_response = chat_with_openai(user_input) print(f"AI says: {ai_response}")

 
speak_with_jaw(ai_response)



if _name_ == "_main_": main()

