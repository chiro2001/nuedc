import time
import os
import cv2


def generate():
    if not os.path.exists("calibration"):
        os.mkdir("calibration")
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow("frame", frame)
        key = cv2.waitKey(0)
        k = chr(key & 0xFF)
        if k == 'y':
            cv2.imwrite(os.path.join("calibration", f"{int(time.time())}.png"), frame)
        elif k == 'q':
            break
    cap.release()


if __name__ == '__main__':
    generate()
