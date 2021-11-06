import os
import cv2
import numpy as np
from capture import start_capture, find_cameras
from ctypes import *
from MvImport.MvCameraControl_class import *
import time
from update_config import update_config
from utils import *
from calc_time import calc_time

last_frame = None
last_time = time.time()
last_timestamp = None
fps_time = []
fps_count = 60

outline_rounds = []
outline_count = 2

states = {
    "big": 0,
    "small": 1,
}
state = "small"


def state_big(frame: np.ndarray, on_quit=None, info=None):
    global last_frame, last_time, fps_time, last_timestamp, outline_rounds
    if last_frame is None:
        last_frame = frame
        return
    if info is not None:
        if last_timestamp is None:
            # last_timestamp = info.nDevTimeStampLow
            last_timestamp = info.nHostTimeStamp
        else:
            # timestamp = info.nDevTimeStampLow
            # timestamp = info.nHostTimeStamp
            timestamp = get_timestamp_ms(info)
            delta_timestamp = timestamp - last_timestamp
            last_timestamp = timestamp
            # print(f"offset: {((timestamp & 0xFFFFC000) >> 14) * 1000 / (2**13)}ms")
            # print(delta_timestamp)
            # print(f"delta: {delta_timestamp}ms, {int(1 / delta_timestamp * 1000)} fps")
    gray = frame
    diff = np.array(np.abs(np.array(gray, dtype=np.int16) -
                           np.array(last_frame, dtype=np.int16)), dtype=np.uint8)
    _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
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
        # for x in bounding_rects[:2]:
        #     x, y, w, h = x[1]
        #     result4 = cv2.rectangle(result4, (x, y), (x + w, y + h), 127, 5)
    else:
        if len(bounding_rects) == 1:
            outline = bounding_rects[0][1]
    if outline is None:
        # print(f"outline is None!")
        return

    result4 = cv2.rectangle(result4, tuple(outline[0:2]), tuple(np.array(outline[0:2]) + np.array(outline[2:4])), 127,
                            5)

    outline_box = [*outline[0:2], *(np.array(outline[0:2]) + np.array(outline[2:4]))]
    width_rate = abs(outline_box[3] - outline_box[0]) / frame.shape[1]
    if width_rate < 0.08:
        # print(f"too thim! {width_rate}")
        # return
        pass
    else:
        # print(f"width_rate = {width_rate}")
        outline_rounds.append(outline_box)
        if len(outline_rounds) >= outline_count:
            outline_rounds = outline_rounds[1:]
    outline_rounds_sum = np.zeros(np.array(outline_box).shape)
    for r in outline_rounds:
        outline_rounds_sum += np.array(r)
    outline_box = outline_rounds_sum / len(outline_rounds)

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

    width = frame.shape[1] // 1
    height = frame.shape[0] // 1
    resized = cv2.resize(result4, (width, height))
    # resized = cv2.resize(frame, (width, height))
    cv2.imshow("frame", resized)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    now = time.time()
    time_delta = now - last_time
    print(
        f"[{frame.shape[1]}x{frame.shape[0]}] calc done! fps: {(sum(fps_time) / fps_count):2.3f} "
        f"fps cnt: {int(1 / time_delta)}")
    fps_time.append(1 / time_delta)
    if len(fps_time) >= fps_count:
        fps_time = fps_time[1:]
    last_time = now

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q' and on_quit is not None:
        on_quit()


Ls = []
Ts_offset = 2
Ls_count = 4


def state_small(frame: np.ndarray, on_quit=None, info=None):
    global Ts_offset, Ls
    res = calc_time(frame, info)
    if res is not None:
        if Ts_offset > 0:
            Ts_offset -= 1
        else:
            T = res
            # g = (((2 * np.pi) / T) ** 2) * 1.056
            # g = 9.764834804943986
            g = 9.7925
            # print(f"g = {g}")
            L = ((T / (2 * np.pi)) ** 2) * g
            print(f"L = {L}")
            Ls.append(L)
            if len(Ls) >= Ls_count:
                ave = np.sum(np.array(Ls)) / len(Ls)
                print(f"ave = {ave}")
                Ls = Ls[1:]


def on_frame(frame: np.ndarray, on_quit=None, info=None, cam=None, on_pause=None):
    if state == 'big':
        cv2.destroyAllWindows()
        update_config(cam, "big", on_pause)
        state_big(frame, on_quit, info)
    elif state == 'small':
        cv2.destroyAllWindows()
        update_config(cam, "small", on_pause)
        state_small(frame, on_quit, info)


def main():
    hostname = os.popen("hostname").readline()
    camera_target = [
        '192.168.137.21',
        '192.168.137.23'
    ]
    camera_id = int(hostname.replace('\n', "")[-1]) - 1
    device_list = find_cameras()
    mvcc_dev_info = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ip_addr = "%d.%d.%d.%d\n" % (((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24),
                                 ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16),
                                 ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8),
                                 (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff))
    if ip_addr == camera_target[0]:
        print(f"using: cam0{camera_id} {camera_target[camera_id]}")
        start_capture(device_list, camera_id, on_frame)
    else:
        print(f"using: cam0{camera_id} {camera_target[camera_id]}")
        start_capture(device_list, 1 - camera_id, on_frame)


if __name__ == '__main__':
    main()
