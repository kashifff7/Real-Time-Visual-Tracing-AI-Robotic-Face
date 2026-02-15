main.py
from eyelid_and_tracking import run_eyelid_and_tracking from jaw_movement import speak_with_jaw
from ai_voice_exchange import record_audio, transcribe_audio, chat_with_openai
import os import threading

def main():
print("Voice Assistant is ready. Press Ctrl+C to stop.")

# Run eyelid and face tracking in a separate thread
tracking_thread = threading.Thread(target=run_eyelid_and_tracking) tracking_thread.start()

# Start initial greeting
speak_with_jaw("Hello! How can I assist you today?")

while True:
# Record user audio
user_audio_path = record_audio(duration=10)

# Transcribe the recorded audio
user_input = transcribe_audio(user_audio_path) print(f"You said: {user_input}")

# Check if the user wants to exit
if user_input.strip().lower() == "exit": speak_with_jaw("Goodbye!") os.remove(user_audio_path)
 
break

# Get AI response
ai_response = chat_with_openai(user_input) print(f"AI says: {ai_response}")

# Speak AI response with jaw movement speak_with_jaw(ai_response)

# Optionally, ask how else the assistant can help speak_with_jaw("How else can I assist you?")

# Clean up the recorded audio file os.remove(user_audio_path)

if name == "main": main()
