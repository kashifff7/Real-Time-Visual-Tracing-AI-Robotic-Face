import time import pyttsx3 import threading import board import busio
from adafruit_pca9685 import PCA9685
from threading import Event # Added for clean termination

# Setup I2C and PCA9685
i2c = busio.I2C(board.SCL, board.SDA) pwm = PCA9685(i2c)
pwm.frequency = 50

RIGHT_JAW_CHANNEL = pwm.channels[6] LEFT_JAW_CHANNEL = pwm.channels[8]

JAW_RIGHT_CLOSED = 1250
JAW_RIGHT_OPEN = 1700
JAW_LEFT_CLOSED = 1700
JAW_LEFT_OPEN = 1150

# Global variables is_speaking = False
stop_jaw_event = Event() # Added for clean termination

def set_servo_pulse_us(channel, pulse_us):
duty = int(pulse_us * 65535 / (1000000 / pwm.frequency)) duty = max(0, min(65535, duty))
channel.duty_cycle = duty

def close_jaw():
for pulse in range(JAW_RIGHT_OPEN, JAW_RIGHT_CLOSED - 1, -25):
 
if stop_jaw_event.is_set(): # Check for termination signal break
set_servo_pulse_us(RIGHT_JAW_CHANNEL, pulse)
mirror = JAW_LEFT_CLOSED - (pulse - JAW_RIGHT_CLOSED) set_servo_pulse_us(LEFT_JAW_CHANNEL, mirror) time.sleep(0.1)

def open_jaw():
for pulse in range(JAW_RIGHT_CLOSED, JAW_RIGHT_OPEN + 1, 25): if stop_jaw_event.is_set(): # Check for termination signal
break set_servo_pulse_us(RIGHT_JAW_CHANNEL, pulse)
mirror = JAW_LEFT_CLOSED - (pulse - JAW_RIGHT_CLOSED) set_servo_pulse_us(LEFT_JAW_CHANNEL, mirror) time.sleep(0.1)

def animate_jaw():
while not stop_jaw_event.is_set() and is_speaking: # Check both flags open_jaw()
time.sleep(0.2) close_jaw() time.sleep(0.2)
def speak_with_jaw(text): global is_speaking is_speaking = True
stop_jaw_event.clear() # Reset event flag jaw_thread = threading.Thread(target=animate_jaw) jaw_thread.start()
engine.say(text) engine.runAndWait() is_speaking = False jaw_thread.join()
def cleanup_jaw(): # Added for program termination stop_jaw_event.set()
close_jaw()

engine = pyttsx3.init()
