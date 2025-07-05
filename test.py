from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array()
    cv2.imshow("Pi Camera", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
picam2.close()
