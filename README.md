# Real-Time Visual Tracing AI Robotic Face

A low-cost robotic face on a Raspberry Pi 4 Model B. Its eyes turn toward the user's face, its eyelids blink, and it answers spoken questions while its jaw opens and closes.

This is the code that ran on the prototype described in the paper "Design and Development of a Low-Cost AI-Based Robotic Face for Real-Time Visual Tracking and Voice Interaction".

## What it does, and what it does not do

- **Face following.** The camera sits inside one eyeball. Each frame (320 × 240) is searched for a face with an OpenCV Haar cascade, and the face position is mapped directly to the eye angles: 60–120° horizontally and 120–60° vertically. No smoothing is applied.
- **Blinking.** The eyelids blink every 10 s while no face is in view.
- **Conversation.** The robot records a fixed 5 s window. Speech is transcribed with the OpenAI whisper-1 API, and if it contains the wake word "Sandy", the question goes to an OpenAI chat model. The reply is spoken with pyttsx3 (eSpeak) on the Pi.
- **Jaw.** While the reply is spoken, the jaw opens for 0.5 s and closes for 0.5 s. This shows that the robot is speaking, but the movement is **not** synchronized with the words or their loudness.
- **Not included.** The head is fixed (no neck or pan–tilt joint), and the face shows no expressions; only the eyes, eyelids and jaw move. The robot follows the user's face but does not measure where the user is looking.

## Hardware

| Part | Details |
| --- | --- |
| Computer | Raspberry Pi 4 Model B, 4 GB, Raspberry Pi OS 64-bit (Bookworm) |
| Camera | Raspberry Pi Camera Module (OV5647), inside the moving eyeball |
| Servo driver | PCA9685 16-channel PWM board on I²C |
| Servos | 6 × SG90 (eyes and eyelids), 1 × MG996R (jaw) |
| Power | Separate 5 V 10 A supply for the servos, grounds joined with the Pi |
| Audio | USB microphone and speaker |

### Wiring

| Pi header pin | PCA9685 |
| --- | --- |
| Pin 1 (3.3 V) | VCC |
| Pin 3 (GPIO 2, SDA) | SDA |
| Pin 5 (GPIO 3, SCL) | SCL |
| Pin 6 (GND) | GND |

The servo supply goes to the PCA9685 V+ terminal.

| PCA9685 channel | Servo |
| --- | --- |
| 0 | Eye, horizontal (SG90) |
| 1 | Eye, vertical (SG90) |
| 2, 3 | Left and right upper eyelid (SG90) |
| 4, 5 | Left and right lower eyelid (SG90) |
| 8 | Jaw (MG996R) |

## Files

- `main.py` starts the vision thread and the conversation loop, and stops both with Ctrl+C.
- `eye_module.py` contains the vision thread: Picamera2 capture, Haar face detection, the eye-angle mapping and blinking.
- `ai_voice_bot.py` contains the conversation loop (recording, speech recognition, chat model, speech output) and the jaw thread.
- `utils/servo_controller.py` sets servo angles through Adafruit ServoKit.

## Running it

```bash
sudo apt install python3-picamera2 python3-opencv espeak
pip install adafruit-circuitpython-servokit sounddevice requests pyttsx3 phonemizer --break-system-packages
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4o-mini   # gpt-3.5-turbo, used on the prototype, has been retired
python3 main.py
```

Say "Sandy" followed by a question within the 5 s recording window.

The API key is read from the environment and is never stored in the code.

## License

MIT License.
