import cv2
import numpy as np
from utils import *

states = {
    "calc_brightness": 0,
    "calc_time_white": 1,
    "calc_time_black": 2,
}
state = "calc_brightness"
brightness_count = 0
g_brightness = None
g_brightness_data = []

timestamp_start = None
time_delta = None
T1 = None
T2 = None

temp = None

brightness_last = None


# 按状态机理解


def calc_brightness(crop, info):
    brightness = np.sum(crop) / crop.size
    # print(f"brightness = {brightness}")
    if state == 'calc_brightness':
        global g_brightness_data
        g_brightness_data.append(brightness)
    return brightness


def calc_time(frame: np.ndarray, info):
    timestamp = get_timestamp_ms(info)
    # print(f"timestamp: {timestamp}")
    target_size = (8, 64)
    size = (frame.shape[1], frame.shape[0])
    crop = frame[((size[1] - target_size[1]) // 2):(((size[1] - target_size[1]) // 2) + target_size[1]),
           ((size[0] - target_size[0]) // 2):(((size[0] - target_size[0]) // 2) + target_size[0])]
    # left = frame[0:(size[1] // 2), :]
    # right = frame[(size[1] // 2):size[1], :]
    # cv2.imshow("small", frame)
    # cv2.waitKey(1)

    global brightness_count, state, g_brightness, brightness_last, time_delta, timestamp_start
    # if state == 'calc_brightness':
    #     brightness = calc_brightness(crop, info)
    #     brightness_count += 1
    #     if brightness_count >= 200:
    #         g_brightness = np.sum(np.array(g_brightness_data)) / len(g_brightness_data)
    #         print(f"g_brightness = {g_brightness}")
    #         if brightness > g_brightness * 0.6:
    #             state = "calc_time_white"
    #         else:
    #             state = "calc_time_black"
    # else:
    #     brightness_left = calc_brightness(left, info)
    #     brightness_right = calc_brightness(right, info)
    #     brightness = calc_brightness(crop, info)
    #     if brightness_last is None:
    #         brightness_last = brightness
    #     if timestamp_start is None:
    #         timestamp_start = timestamp
    #     if brightness_left > brightness_right:
    #         if brightness_last > brightness:
    #             if time_delta is None:
    #                 time_delta = timestamp - timestamp_start
    #                 timestamp_start = timestamp
    #                 print(f"T = {time_delta}")
    #         else:
    #             time_delta = None
    #
    #     brightness_last = brightness

    if state == 'calc_brightness':
        brightness = calc_brightness(crop, info)
        brightness_count += 1
        if brightness_count >= 200:
            g_brightness = np.sum(np.array(g_brightness_data)) / len(g_brightness_data)
            print(f"g_brightness = {g_brightness}")
            if brightness > g_brightness * 0.6:
                state = "calc_time_white"
            else:
                state = "calc_time_black"
    else:
        global timestamp_start, time_delta, T1, T2
        brightness = calc_brightness(crop, info)
        if state == 'calc_time_white':
            if brightness > 0.5 * g_brightness:
                if timestamp_start is None:
                    timestamp_start = timestamp
                else:
                    if time_delta is None:
                        time_delta = timestamp - timestamp_start
                        timestamp_start = None
                        # if T1 is None:
                        #     T1 = time_delta / 1000
                        #     T = T1
                        #     T1 = None
                        #     print(f"T = {T}")
                        if T1 is None:
                            T1 = time_delta / 1000
                            # print(f"T1 = {T1}")
                        else:
                            if T2 is None:
                                T2 = time_delta / 1000
                                # print(f"T2 = {T2}")
                            else:
                                T = T1 + T2
                                T1 = None
                                T2 = None
                                print(f"T = {T}s")
                                return T
                    else:
                        pass
            else:
                time_delta = None
                # print(f"-> black")
                state = "calc_time_black"
        else:
            if brightness > 0.5 * g_brightness:
                # print(f"-> white")
                state = "calc_time_white"
            else:
                pass
