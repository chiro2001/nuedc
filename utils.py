import time


# last = 0


def get_timestamp_ms(info):
    # 摄像头的硬件计时不准确，吧不知道为什么
    # global last
    # return ((info.nHostTimeStamp & 0xFFFFC000) >> 14) * 1000 / (2 ** 13)
    # print(
    #     f"{(info.nHostTimeStamp & 0xF8000000) >> 27}:{(info.nHostTimeStamp & 0x7FFC000) >> 14}:{(info.nHostTimeStamp & 0x3FFF) >> 0}")
    # ms = (info.nHostTimeStamp & 0x7FFC000) >> 14
    # s = (info.nHostTimeStamp & 0xF8000000) >> 27
    # print(f"[{(s + ms / 8192):.3f}] {s}:{ms}")
    # return s + ms / 1000
    # res = (s + ms / 8192) * 1000
    # res = time.time() * 1000
    # fps = 1 / ((res - last) / 1000)
    # print(f"fps = {fps}")
    # last = res
    # return res

    # 用软件计时会产生误差和抖动，务必小心
    return time.time() * 1000
