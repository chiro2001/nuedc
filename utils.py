import time
last = 0


def get_timestamp_ms(info):
    global last
    # return ((info.nHostTimeStamp & 0xFFFFC000) >> 14) * 1000 / (2 ** 13)
    # print(
    #     f"{(info.nHostTimeStamp & 0xF8000000) >> 27}:{(info.nHostTimeStamp & 0x7FFC000) >> 14}:{(info.nHostTimeStamp & 0x3FFF) >> 0}")
    ms = (info.nHostTimeStamp & 0x7FFC000) >> 14
    s = (info.nHostTimeStamp & 0xF8000000) >> 27
    # print(f"[{(s + ms / 8192):.3f}] {s}:{ms}")
    # return s + ms / 1000
    # res = (s + ms / 8192) * 1000
    res = time.time() * 1000
    # fps = 1 / ((res - last) / 1000)
    # print(f"fps = {fps}")
    last = res
    return res
