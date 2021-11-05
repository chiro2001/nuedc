import cv2
import numpy as np
from capture import start_capture, find_cameras

last_frame = None


def on_frame(frame: np.ndarray, on_quit=None):
    global last_frame
    if last_frame is None:
        last_frame = frame
        return
    gray = frame
    diff = np.array(np.abs(np.array(gray, dtype=np.int16) -
                           np.array(last_frame, dtype=np.int16)), dtype=np.uint8)
    _, threshold = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image_open = cv2.morphologyEx(
        threshold, cv2.MORPH_OPEN, kernel=kernel_open)
    contours, hierarchy = cv2.findContours(
        image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_rects = [(c, cv2.boundingRect(c)) for c in contours]
    list.sort(bounding_rects, key=lambda x: x[1][2] + x[1][3], reverse=True)
    # result = np.zeros(gray.shape, dtype=gray.dtype)
    # result = cv2.drawContours(
    #     result, [x[0] for x in bounding_rects[:2]], -1, 255, 3)
    # result4 = np.zeros(gray.shape, dtype=gray.dtype)
    result4 = gray.copy()
    outline = None
    if len(bounding_rects) >= 2:
        outline = [min(bounding_rects[0][1][0], bounding_rects[1][1][0]),
                   min(bounding_rects[0][1][1], bounding_rects[1][1][1]),
                   max(bounding_rects[0][1][2], bounding_rects[1][1][2]),
                   max(bounding_rects[0][1][3], bounding_rects[1][1][3])]
        for x in bounding_rects[:2]:
            x, y, w, h = x[1]
            result4 = cv2.rectangle(result4, (x, y), (x + w, y + h), 127, 5)
    else:
        if len(bounding_rects) == 1:
            outline = bounding_rects[0][1]
    if outline is None:
        # print(f"outline is None!")
        return

    result4 = cv2.rectangle(result4, tuple(outline[0:2]), tuple(np.array(outline[0:2]) + np.array(outline[2:4])), 127,
                            5)

    outline_box = [*outline[0:2], *(np.array(outline[0:2]) + np.array(outline[2:4]))]

    # plot = gray.copy()
    center = tuple(map(int, [(outline_box[0] + outline_box[2]) / 2, (outline_box[1] + outline_box[3]) / 2]))
    pts = [
        ((outline_box[0] + outline_box[2]) / 2, outline_box[1]),
        ((outline_box[0] + outline_box[2]) / 2, outline_box[3])
    ]
    cv2.circle(result4, center, 5, 255, -1)
    for p in pts:
        cv2.circle(result4, tuple(map(int, p)), 3, 255, -1)

    last_frame = gray
    # cv2.imshow("result4", result4)
    # cv2.imshow("result", result)

    width = frame.shape[1] // 5
    height = frame.shape[0] // 5
    resized = cv2.resize(result4, (width, height))
    # resized = frame
    cv2.imshow("frame", resized)
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q' and on_quit is not None:
        on_quit()


def main():
    device_list = find_cameras()
    start_capture(device_list, 0, on_frame)


if __name__ == '__main__':
    main()
