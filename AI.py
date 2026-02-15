ai_voice_exchange.py import requests
import sounddevice as sd import wave
import os import time

# OpenAI API setup API_KEY = "api_key_here"
WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions" GPT_URL = "https://api.openai.com/v1/chat/completions"

# Function to record audio
def record_audio(duration=10, samplerate=16000): print(f"Recording for {duration} seconds...")
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
sd.wait()
print("Recording complete.") file_path = "user_audio.wav"
with wave.open(file_path, 'wb') as wf: wf.setnchannels(1) wf.setsampwidth(2) wf.setframerate(samplerate) wf.writeframes(audio_data.tobytes())
 
return file_path

# Function to transcribe audio using OpenAI's Whisper API def transcribe_audio(file_path):
try:
print("Uploading audio for transcription...")
headers = {"Authorization": f"Bearer {API_KEY}"} files = {
"file": (file_path, open(file_path, "rb")), "model": (None, "whisper-1"),
}
response = requests.post(WHISPER_URL, headers=headers, files=files) response.raise_for_status()
result = response.json() print("Transcription complete!") return result.get("text", "")
except requests.exceptions.RequestException as e: print(f"Error during transcription: {e}")
return "Could not transcribe audio."

# Function to interact with OpenAI GPT and generate a response def chat_with_openai(prompt):
try:
print("Sending message to GPT...")
headers = {"Authorization": f"Bearer {API_KEY}"} json_data = {
"model": "gpt-3.5-turbo", "messages": [
{"role": "system", "content": "You are a helpful voice assistant."},
{"role": "user", "content": prompt},
],
"temperature": 0.7,
}
response = requests.post(GPT_URL, headers=headers, json=json_data) response.raise_for_status()
result = response.json()
return result["choices"][0]["message"]["content"]
 
except requests.exceptions.RequestException as e: print(f"OpenAI Chat failed: {e}")
return "Sorry, I couldn't process that."
