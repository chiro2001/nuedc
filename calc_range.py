import cv2
import numpy as np

raw_image = None
superposition = None


def init_superposition(shape=None):
    global superposition
    if shape is None:
        shape = raw_image.shape
    superposition = np.zeros(shape, dtype=raw_image.dtype)


def set_raw_image(frame):
    global raw_image, superposition
    raw_image = frame.copy()
    init_superposition()


def add_frame(frame):
    global superposition
    try:
        diff = np.array(np.abs(np.array(frame, dtype=np.int16) -
                               np.array(raw_image, dtype=np.int16)), dtype=np.uint8)
        _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        diff = threshold
    except Exception as e:
        print(f"add_frame: {e}")
        set_raw_image(frame)
        diff = np.array(np.abs(np.array(frame, dtype=np.int16) -
                               np.array(raw_image, dtype=np.int16)), dtype=np.uint8)
    superposition = cv2.bitwise_or(superposition, diff)


def calc_range():
    image: np.ndarray = superposition.copy()
    sums = np.array([np.sum(line) for line in image.T], dtype=np.float)
    sums -= np.min(sums)
    m = np.max(sums) if np.max(sums) != 0 else 1
    sums /= m
    left = 0
    right = len(sums) - 1
    while left < len(sums) and sums[left] < 0.45:
        left += 1
    while right >= 0 and sums[right] < 0.45:
        right -= 1
    ans = right - left
    ans = max(ans, 0)
    # print(f"[{right} - {left} = {ans}]sums: {sums.shape}")
    cv2.imshow("superposition", image)
    cv2.waitKey(1)
    return ans
