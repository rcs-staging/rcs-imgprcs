import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import (
    GestureRecognizer, GestureRecognizerOptions,
    RunningMode
)
from picamera2 import Picamera2
import cv2
import math
import serial
import time

# UART Setup
arduino_port = '/dev/ttyUSB0'
arduino = serial.Serial(arduino_port, 9600, timeout=1)

# Path to MediaPipe gesture model
MODEL_ASSET_PATH = "gesture_recognizer.task"

action_text = ""
last_command = ""

ACTION_LABELS = {
    "F": "Forward",
    "B": "Backward",
    "S": "Stop"
}

def start_camera():
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={'size': (640, 480), 'format': 'RGB888'},
    )
    picam.configure(config)
    picam.start()
    return picam

def capture_frame(cam):
    frame = cam.capture_array("main")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, Image(data=rgb, image_format=ImageFormat.SRGB)

def estimate_hand_distance_cm(landmarks):
    # Estimate distance between wrist (0) and middle fingertip (12)
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    dx = middle_tip.x - wrist.x
    dy = middle_tip.y - wrist.y
    pixel_dist = math.sqrt(dx * dx + dy * dy)

    if pixel_dist == 0:
        return 1000  # Invalid case fallback

    # Approximate scale: adjust based on real calibration
    distance_cm = 0.12 / pixel_dist * 50
    return distance_cm

def draw_hand_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    landmark_points = []
    for lm in hand_landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        landmark_points.append((x_px, y_px))
        cv2.circle(frame, (x_px, y_px), 5, (0, 0, 255), -1)
    for start_idx, end_idx in mp.solutions.hands.HAND_CONNECTIONS:
        start = landmark_points[start_idx]
        end = landmark_points[end_idx]
        cv2.line(frame, start, end, (255, 255, 255), 2)

def send_command_to_arduino(cmd):
    global last_command
    if cmd != last_command:
        arduino.write((cmd + "\n").encode())
        print(f"Sent to Arduino: {cmd}")
        last_command = cmd

def run_recognition():
    global action_text
    cam = start_camera()

    options = GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_ASSET_PATH),
        running_mode=RunningMode.IMAGE
    )

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cam.is_open:
            frame_bgr, mp_image = capture_frame(cam)
            frame_copy = frame_bgr.copy()
            result = recognizer.recognize(mp_image)

            if result.hand_landmarks:
                # At least one hand detected
                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(frame_copy, hand_landmarks)

                    # Estimate distance
                    distance_cm = estimate_hand_distance_cm(hand_landmarks)
                    cv2.putText(frame_copy, f"Distance: {int(distance_cm)}cm", (10, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Determine action
                    if distance_cm > 20:
                        action_text = "F"
                    elif distance_cm < 18:
                        action_text = "B"
                    else:
                        action_text = "S"

                    send_command_to_arduino(action_text)

            else:
                # No hand detected
                action_text = "S"
                send_command_to_arduino(action_text)

            # Display action
            cv2.putText(frame_copy, f"Action: {ACTION_LABELS.get(action_text, 'Unknown')}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            cv2.imshow("Distance-Based Control", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.close()
    cv2.destroyAllWindows()
    arduino.close()


if __name__ == "__main__":
    run_recognition()
