# Install Python dependencies.
python3 -m pip install pip --upgrade --break-system-packages
python3 -m pip install -r requirements.txt --break-system-packages

wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
