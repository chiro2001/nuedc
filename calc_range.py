import cv2
import numpy as np

raw_image = None
superposition = None


def init_superposition():
    global superposition
    superposition = np.zeros(raw_image.shape, dtype=raw_image.dtype)


def set_raw_image(frame):
    global raw_image, superposition
    raw_image = frame.copy()
    init_superposition()


def add_frame(frame):
    global superposition
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
    while sums[left] < 0.5 and left < len(sums):
        left += 1
    while sums[right] < 0.5 and right >= 0:
        right -= 1
    ans = right - left
    # print(f"[{right} - {left} = {ans}]sums: {sums.shape}")
    cv2.imshow("image", image)
    cv2.waitKey(1)
    return ans
