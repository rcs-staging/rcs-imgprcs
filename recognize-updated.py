import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
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

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
action_text = ""
last_command = ""

# Optional: To display full label on screen
ACTION_LABELS = {
    "F": "Forward",
    "B": "Backward",
    "R": "Right",
    "L": "Left",
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

def get_angle(a, b, c):
    ab = [a.x - b.x, a.y - b.y]
    cb = [c.x - b.x, c.y - b.y]
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    ab_len = math.hypot(*ab)
    cb_len = math.hypot(*cb)
    if ab_len * cb_len == 0:
        return 0
    angle = math.acos(dot / (ab_len * cb_len))
    return math.degrees(angle)

def is_finger_up_by_angle(hand_landmarks, handedness_label):
    lm = hand_landmarks
    up = []

    # --- Thumb Detection
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]
    wrist = lm[0]

    angle_thumb = get_angle(thumb_tip, thumb_ip, thumb_mcp)
    dist_tip_wrist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    dist_mcp_wrist = math.hypot(thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y)

    thumb_up = angle_thumb > 150 and dist_tip_wrist > dist_mcp_wrist
    up.append(thumb_up)

    # --- Other Fingers
    for i in range(1, 5):
        tip = lm[FINGER_TIPS[i]]
        pip = lm[FINGER_PIPS[i]]
        mcp = lm[FINGER_MCPS[i]]
        angle = get_angle(tip, pip, mcp)
        up.append(angle > 160)

    return up

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

def draw_finger_states(frame, states):
    cv2.putText(frame, "Hand:", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for i, is_up in enumerate(states):
        txt = f"  {FINGER_NAMES[i]}: {'Up' if is_up else 'Down'}"
        cv2.putText(frame, txt, (10, 80 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if is_up else (0, 0, 255), 2)

def map_fingers_to_action(count):
    if count == 1:
        return "F"  # Forward
    elif count == 2:
        return "B"  # Backward
    elif count == 3:
        return "R"  # Right
    elif count == 4:
        return "L"  # Left
    else:  # 0 or 5 fingers
        return "S"  # Stop

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
            action_text = "S"  # Default to Stop

            if result.hand_landmarks and result.handedness:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    handed_label = result.handedness[i][0].category_name
                    fingers = is_finger_up_by_angle(hand_landmarks, handed_label)
                    num_up = sum(fingers)
                    action_text = map_fingers_to_action(num_up)

                    draw_hand_landmarks(frame_copy, hand_landmarks)
                    draw_finger_states(frame_copy, fingers)
                    send_command_to_arduino(action_text)

            # Display action on screen
            cv2.putText(frame_copy, f"Action: {ACTION_LABELS.get(action_text, 'Unknown')}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            cv2.imshow("Finger Detection", frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.close()
    cv2.destroyAllWindows()
    arduino.close()

if __name__ == "__main__":
    run_recognition()
