import os
import cv2
import json
import threading
import base64
import numpy as np
from capture import start_capture, find_cameras, update_buf
from ctypes import *
from MvImport.MvCameraControl_class import *
import time
from update_config import update_config
from utils import *
import io
from calc_time import calc_time
from calc_range import calc_range, add_frame, set_raw_image, init_superposition

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

import xmlrpc.client

# 全局变量

# 计算fps用
last_frame = None
last_time = time.time()
last_timestamp = None
fps_time = []
fps_count = 60

# 框红框位置，滤波用
outline_rounds = []
outline_count = 2

# 测量角度结果
D_res = None
# 测量角度至少需要多少帧
D_res_count = 20

states = {
    "init": 0,
    "big": 1,
    "small": 2
}
# 当前系统状态
state = "init"

# 测量长度结果数据记录
Ls = []
# 抛弃多少个开头的测量结果
Ts_offset = 3
# 计算多少个平均值
Ls_count = 2
# 长度调整系数（单位：m）
L_delta = 5.37 / 100
# 计算角度的像素宽度调整（激光笔宽度）
D_delta = 40
# 求得结果
L_result = None
# 结果排名（未用到）
L_rank = 0

# 上一个框框
last_outline = None

# 全局帧（用来图传）
g_frame = None
# 接收到B的图像
g_slave_frame = None
# C接收到A、B的图像
g_display_A = None
g_display_B = None

# 状态切换是否完成
switched = False

# 系统闲置状态开始时间
idle_start = None
# 系统闲置状态总时间（未用）
idle_time = 5

# 是否等待input输入
blocked = True
# 按键输入线程
th_wait_key = None

# 指定主机
master = os.environ.get("MASTER", "rpi01")
# 指定终端
display = os.environ.get("DISP", "rpi00")
# 用hostname定位身份
myself = os.popen("hostname").readline().replace("\n", "")
is_master = master == myself
is_display = display == myself
if is_master:
    print(f"Master running!")
else:
    print(f"Slave running!")

# 终端显示结果用
g_display_result = None
g_display_state = "init"
# 终端屏幕覆盖层大小（屏幕大小）
g_display_size = (1920, 1080)
g_display_font = cv2.FONT_HERSHEY_SIMPLEX

# A从B拿到的数据
slave_D_res = None
slave_L_res = None
slave_L_rank = None

# RPC调用服务器（B和C）
server = None
server_display = None

# 等待长度和角度数据的限制时间
timeout_L = 20
timeout_D = 20


# 得到增强了对比度的图像
def get_enhanced_frame(frame, alpha=1.3, beta=30):
    if frame is None:
        return None
    # TODO: fix uint8 溢出
    return np.uint8(np.clip((alpha * frame + beta), 0, 255))


# 得到缩小的图像
def get_resized_frame(frame):
    if frame is None:
        return None
    width = frame.shape[1] // 5
    height = frame.shape[0] // 5
    resized = cv2.resize(frame, (width, height))
    return resized


# 得到放大图像
def get_expanded_frame(frame):
    if frame is None:
        return None
    width = int(frame.shape[1] * 5)
    height = int(frame.shape[0] * 5)
    resized = cv2.resize(frame, (width, height))
    return resized


# 框柱运动物体
def box_frame(frame):
    global last_frame, last_time, fps_time, last_timestamp, outline_rounds
    resized_raw = get_resized_frame(frame)
    if last_frame is None:
        last_frame = frame.copy()
        print("first_frame!")
        return resized_raw

    gray = frame
    try:
        diff = np.array(np.abs(np.array(gray, dtype=np.int16) -
                               np.array(last_frame, dtype=np.int16)), dtype=np.uint8)
    except Exception as e:
        print(f"state_big: {e}")
        last_frame = gray.copy()
        diff = np.array(np.abs(np.array(gray, dtype=np.int16) -
                               np.array(last_frame, dtype=np.int16)), dtype=np.uint8)
    _, threshold = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_open = cv2.morphologyEx(
        threshold, cv2.MORPH_OPEN, kernel=kernel_open)
    contours, hierarchy = cv2.findContours(
        image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_rects = [(c, cv2.boundingRect(c)) for c in contours]
    list.sort(bounding_rects, key=lambda x: x[1][2] + x[1][3], reverse=True)
    # result4 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    result4 = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2RGB)
    # result4 = cv2.cvtColor(image_open, cv2.COLOR_GRAY2RGB)
    outline = None
    global last_outline
    if len(bounding_rects) >= 2:
        f = [min(bounding_rects[0][1][0], bounding_rects[1][1][0]),
             min(bounding_rects[0][1][1], bounding_rects[1][1][1])]
        outline = [*f,
                   max(bounding_rects[0][1][2] + f[0], bounding_rects[1][1][2] + f[0]),
                   max(bounding_rects[0][1][3] + f[1], bounding_rects[1][1][3] + f[1])]
        for b in bounding_rects:
            outline = [min(outline[0], b[1][0]),
                       min(outline[1], b[1][1]),
                       max(outline[2], b[1][2] + b[1][0]),
                       max(outline[3], b[1][3] + b[1][1])]
    else:
        if len(bounding_rects) == 1:
            outline = bounding_rects[0][1]
    if outline is None:
        # print(f"outline is None!")
        # return resized_raw
        outline = last_outline
    if outline is None:
        print(f"outline is None!")
        return resized_raw

    # result4 = cv2.rectangle(result4, tuple(outline[0:2]), tuple(np.array(outline[0:2]) + np.array(outline[2:4])),
    #                         (0, 0, 255), 5)

    last_outline = outline
    result4 = cv2.rectangle(result4, tuple(outline[0:2]), tuple(np.array(outline[2:4])), (0, 0, 255), 5)

    last_frame = frame.copy()

    global g_frame
    resized = get_resized_frame(result4)
    g_frame = resized.copy()

    return resized


# 测量角度时候的on_frame
def state_big(frame: np.ndarray, on_quit=None, info=None):
    global last_frame, last_time, fps_time, last_timestamp, outline_rounds
    add_frame(frame)
    ans_range = calc_range()
    global D_res, D_res_count
    print(f"ans_range: {ans_range}, D_res_count: {D_res_count}")
    if ans_range > 0:
        D_res_count -= 1
        if D_res_count <= 0:
            D_res = ans_range
            print(f"D_res: {ans_range}")
    resized = box_frame(frame)
    expanded = get_expanded_frame(resized)
    cv2.imshow("boxed", expanded)
    cv2.setWindowProperty("boxed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    now = time.time()
    time_delta = now - last_time
    print(
        f"[{frame.shape[1]}x{frame.shape[0]}] calc done! fps: {(sum(fps_time) / fps_count):2.3f} "
        f"fps cnt: {int(1 / time_delta)}")
    fps_time.append(1 / time_delta)
    if len(fps_time) >= fps_count:
        fps_time = fps_time[1:]
    last_time = now


# RPC参数
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', '/RPC3')


# 调用改变系统状态
def remote_set_state(s: str):
    global state, switched
    state = s
    switched = False

    if state == 'idle':
        to_idle_state()


# 测量周期的时候的on_frame
def state_small(frame: np.ndarray, on_quit=None, info=None):
    global Ts_offset, Ls, switched, state
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
            L = ((T / (2 * np.pi)) ** 2) * g - L_delta
            print(f"L = {L}")
            Ls.append(L)
            if len(Ls) >= Ls_count:
                # # 删除一个不符合的数据
                # max_d = 0
                # select_index = None
                # ave_raw = float(np.sum(np.array(Ls)) / len(Ls))
                # for i in range(len(Ls)):
                #     m = abs(ave_raw - Ls[i])
                #     if m > max_d:
                #         select_index = i
                #         print(f"del: [{i}]")
                # if select_index is not None:
                #     Ls = [Ls[i] for i in range(len(Ls)) if i != select_index]
                print(f"Ls = {Ls}")
                ave = np.sum(np.array(Ls)) / len(Ls)
                global L_result
                L_result = float(ave)
                print(f"ave = {ave}")
                if is_master:
                    state = 'big'
                    switched = False
                Ls = Ls[1:]


# 单独开线程来等待按键
def wait_key(words: str = "Press Key to continue..."):
    global blocked
    blocked = True
    input(words)
    # print(f"{words}")
    # time.sleep(1)
    blocked = False


# 切换到闲置模式（尝试恢复系统状态）
# TODO: fix 回到 idle 则摄像头配置无法切换
def to_idle_state():
    print(f"[ to_idle_state() ]")
    global state, idle_start
    global D_res, D_res_count
    D_res = None
    D_res_count = 20
    global L_rank, Ls, L_result, Ls_count
    Ls = []
    Ls_count = 2
    L_result = None
    L_rank = 0

    init_superposition()

    state = "idle"
    idle_start = time.time()
    print(f"======= IDLE =======")
    print(f"PLEASE SHAKE IT")
    if is_master:
        global th_wait_key
        th_wait_key = threading.Thread(target=wait_key)
        th_wait_key.start()


# 帧到来事件
def on_frame(frame: np.ndarray, on_quit=None, info=None, cam=None, on_pause=None):
    global switched, state, idle_start
    if state == 'init':
        # 初始化状态，仅仅一帧
        print(f" [ STATE : {state} ]")

        if is_master:
            if server_display is None:
                time.sleep(0.5)
                print(f"======== WA : No C Client, wait... ========")
                return
            if server_display is not None:
                sent = False
                time.sleep(0.5)
                while not sent:
                    try:
                        server_display.set_display_state("init")
                        sent = True
                    except Exception as e:
                        print(f"cannot send result: {e}")
                        time.sleep(0.5)
            else:
                print(f"======== WA : No C Client ========")
        time.sleep(2)
        ok = False
        while not ok:
            try:
                update_config(cam, "big", on_pause)
                ok = True
            except Exception:
                time.sleep(2)
        time.sleep(1)
        update_buf(cam)
        # 更新测量角度的时候的背景图，最好拿掉激光笔再存
        set_raw_image(frame)
        to_idle_state()
        if is_master:
            if server_display is not None:
                sent = False
                time.sleep(0.5)
                while not sent:
                    try:
                        server_display.set_display_state("idle")
                        sent = True
                    except Exception as e:
                        print(f"cannot send result: {e}")
                        time.sleep(0.5)
            else:
                print(f"======== WA : No C Client ========")
    elif state == "idle":
        # 闲置模式
        global g_slave_frame, g_frame
        boxed = box_frame(frame)
        boxed = get_enhanced_frame(boxed, alpha=4, beta=0)
        boxed_expanded = get_expanded_frame(boxed)
        cv2.imshow("boxed", boxed_expanded)
        cv2.setWindowProperty("boxed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        if is_master:
            if server_display:
                try:
                    b64 = encode_b64_img(boxed)
                    server_display.set_display_frame_A(b64)
                except Exception as e:
                    print(f"sending g frame err: {e}")
                try:
                    b64_slave = server.get_g_frame()
                    global g_slave_frame
                    if b64_slave is not None:
                        g_slave_frame = parse_b64_img(b64_slave)
                    g_slave_frame = get_enhanced_frame(g_slave_frame, alpha=4, beta=0)
                    server_display.set_display_frame_B(encode_b64_img(g_slave_frame))
                except Exception as e:
                    print(f"sending frame err: {e}")
                time.sleep(0.1)

            if not blocked:
                server.remote_set_state("small")
                remote_set_state("small")
                print(f"======= L =======")
                print(f"Measuring L...")
    elif state == 'big':
        # 测量角度模式
        if not switched:
            time.sleep(1)
            ok = False
            while not ok:
                try:
                    update_config(cam, "big", on_pause)
                    ok = True
                except Exception:
                    time.sleep(2)
            update_buf(cam)
            switched = True
        state_big(frame, on_quit, info)
    elif state == 'small':
        # 测量长度模式
        if not switched:
            time.sleep(1)
            ok = False
            while not ok:
                try:
                    update_config(cam, "small", on_pause)
                    ok = True
                except Exception:
                    time.sleep(2)
            update_buf(cam)
            switched = True
        state_small(frame, on_quit, info)
    elif state == 'exit':
        sys.exit(0)


# 主线程
def master_thread():
    global slave_L_res, slave_D_res, slave_L_rank
    global state, switched
    # B 必须已经启动
    while server is None:
        time.sleep(0.2)
    server_ok = False
    while not server_ok:
        try:
            server.get_L_rank()
            server_ok = True
        except ConnectionRefusedError:
            time.sleep(0.2)
    while True:
        if state == 'idle':
            time.sleep(0.1)
        elif state == 'small' or state == 'big':
            if server_display is not None:
                try:
                    server_display.set_display_state("measuring")
                except Exception as e:
                    print(f"disp result err: {e}")
                    time.sleep(0.4)
            time_L = 0.0
            time_D = 0.0
            time_d = 0.4
            print(f"Waiting L slave...")
            while slave_L_res is None:
                try:
                    slave_L_rank = server.get_L_rank()
                    slave_L_res = server.get_L_result()
                except Exception as e:
                    print(f"get slave L err: {e}")
                time.sleep(time_d)
                time_L += time_d
                if time_L > timeout_L:
                    print(f"WA: slave_L timeout")
                    break
            time_L = 0.0
            print(f"Waiting L...")
            while L_result is None and slave_L_res is None:
                time.sleep(time_d)
                time_L += time_d
                if time_L > timeout_L:
                    print(f"WA: L timeout")
                    break
            if slave_L_rank is None:
                slave_L_rank = 999
            global L_rank
            if L_rank is None:
                L_rank = 999
            final_result_L = None
            if L_result is not None:
                final_result_L = L_result
                print(f"L: use master")
            elif slave_L_res is not None:
                final_result_L = slave_L_res
                print(f"L: use slave")
            if final_result_L is None:
                if L_rank <= slave_L_rank:
                    # 更倾向用master的（
                    final_result_L = L_result
                    print(f"L: use master")
                else:
                    final_result_L = slave_L_res
                    print(f"L: use slave")
            print(f"final_result_L = {final_result_L}")
            server.remote_set_state("big")
            time.sleep(1)
            remote_set_state('big')
            time.sleep(0.2)
            print(f"Waiting D slave...")
            while slave_D_res is None:
                try:
                    slave_D_res = server.get_D_res()
                except Exception as e:
                    print(f"get slave D err: {e}")
                time_D += time_d
                time.sleep(time_d)
                time_D += time_d
                if time_D > timeout_D:
                    print(f"WA: D slave timeout")
                    break
            time_D = 0.0
            global D_res
            print(f"Waiting D...")
            while D_res is None:
                time_D += time_d
                time.sleep(time_d)
                time_D += time_d
                if time_D > timeout_D:
                    print(f"WA: D timeout")
                    break

            Theta = None
            if slave_D_res is not None:
                slave_D = slave_D_res if slave_D_res != 0 else 1e-9
                D = D_res if D_res is not None else 0
                if D > 40:
                    D = abs(D - D_delta)
                if slave_D > 40:
                    slave_D = abs(slave_D - D_delta)
                theta = np.arctan(D / slave_D)
                Theta = theta / np.pi / 2 * 360
                print(f"theta: {theta} ({Theta})")

            try:
                server.remote_set_state("idle")
            except Exception as e:
                print(f"set to remote idle error: {e}")
            remote_set_state("idle")
            time.sleep(3)

            print(f"\n==================== DONE ==================\n")
            result = {
                'L': float(final_result_L if final_result_L is not None else 0),
                'theta': float(Theta if Theta is not None else 0)
            }
            # 这里保证数据传到C
            if server_display is not None:
                sent = False
                while not sent:
                    try:
                        server_display.display_result(json.dumps(result))
                        server_display.set_display_state("idle")
                        sent = True
                    except Exception as e:
                        print(f"disp result err: {e}")
                        time.sleep(0.4)
            time.sleep(3)


# PRC调用中不能直接传图像，转成base64再传
def parse_b64_img(b64: str):
    if b64 is None:
        return None
    io_buf = io.BytesIO(base64.b64decode(b64))
    decode_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)
    return decode_img


def encode_b64_img(img):
    if img is None:
        return None
    is_success, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode()
    return b64


# 打开PRC服务器（B、C）
def start_rpc_server():
    global server
    server = SimpleXMLRPCServer(("0.0.0.0", 8000), requestHandler=RequestHandler, allow_none=True)
    server.register_introspection_functions()
    server.register_function(remote_set_state)

    # ping-pong
    @server.register_function
    def test():
        return 'OK'

    @server.register_function
    def get_L_rank():
        return L_rank

    @server.register_function
    def get_L_result():
        return L_result

    @server.register_function
    def get_D_res():
        return D_res

    # 勿用
    @server.register_function
    def exit_slave():
        sys.exit(0)

    @server.register_function
    def get_g_frame():
        if g_frame is None:
            return None
        b64 = encode_b64_img(g_frame)
        return b64

    @server.register_function
    def display_result(text_):
        print(f"\n================ RESULT =================\n")
        print(f"{text_}")

        if text_ is None:
            return
        # 完成后由C播放提示音
        os.system("aplay Ring06.wav &")

        global g_display_result
        g_display_result = json.loads(text_)

        return "OK"

    global g_display_A, g_display_B

    @server.register_function
    def set_display_frame_A(b64: str):
        global g_display_A
        display_A = parse_b64_img(b64)
        g_display_A = get_expanded_frame(display_A)
        print(
            f"got disp A [{g_display_A.shape if g_display_A is not None else None}] ({len(b64) if b64 is not None else None})")

    @server.register_function
    def set_display_frame_B(b64):
        global g_display_B
        display_B = parse_b64_img(b64)
        g_display_B = get_expanded_frame(display_B)
        print(
            f"got disp B [{g_display_B.shape if g_display_B is not None else None}] ({len(b64) if b64 is not None else None})")

    @server.register_function
    def set_display_state(s: str):
        global g_display_state
        g_display_state = s

    server.serve_forever()


def destroy_window(name: str):
    try:
        cv2.destroyWindow(name)
    except Exception:
        pass


def create_bg(text):
    bg = np.zeros([g_display_size[1], g_display_size[0], 3], dtype=np.uint8)
    cv2.putText(bg, text, (20, 100), g_display_font, 3, (0, 255, 255), 3)
    return bg


def display_loop():
    while True:
        try:
            if g_display_state == "init":
                bg = create_bg("INIT")
                cv2.imshow("init", bg)
                cv2.waitKey(1)
                cv2.setWindowProperty("init", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                time.sleep(0.02)
            elif g_display_state == "idle":
                destroy_window("measuring")
                destroy_window("init")
                if g_display_result is not None:
                    L, theta = g_display_result.get("L", 0), g_display_result.get("theta", 0)
                    if L is None:
                        L = 0
                    if theta is None:
                        theta = 0
                    res_img = np.zeros((180, 768, 3), dtype=np.uint8)
                    cv2.putText(res_img, f"L: {L:.3f}m", (20, 48), g_display_font, 2, (255, 255, 0), 2)
                    cv2.putText(res_img, f"theta: {theta:.1f}", (20, 120), g_display_font, 2, (255, 255, 0), 2)
                    cv2.putText(res_img, f"DONE", (400, 84), g_display_font, 3, (0, 255, 0), 3)
                    cv2.imshow("result", res_img)
                    cv2.waitKey(1)
                if g_display_A is not None:
                    A = g_display_A.copy()
                    cv2.putText(A, "A", (20, 100), g_display_font, 3, (127, 0, 127), 3)
                    cv2.imshow("A", A)
                    cv2.waitKey(1)
                    time.sleep(0.02)
                else:
                    # print(f"WA: No disp A")
                    pass
                if g_display_B is not None:
                    B = g_display_B.copy()
                    cv2.putText(B, "B", (20, 100), g_display_font, 3, (127, 0, 127), 3)
                    cv2.imshow("B", B)
                    cv2.waitKey(1)
                    time.sleep(0.02)
                else:
                    # print(f"WA: No disp B")
                    pass
            elif g_display_state == 'measuring':
                bg = create_bg("TESTING")
                cv2.imshow("measuring", bg)
                cv2.waitKey(1)
                cv2.setWindowProperty("measuring", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            time.sleep(0.02)
        except Exception as e:
            print(f"display loop: {e}")


# 主程序
def main():
    global server
    if is_display:
        # 如果是C终端，只等待RPC调用
        threading.Thread(target=display_loop, daemon=True).start()
        start_rpc_server()
    else:
        hostname = os.popen("hostname").readline()
        camera_target = [
            '192.168.137.21',
            '192.168.137.23'
        ]
        camera_id = int(hostname.replace('\n', "")[-1]) - 1
        host_ips = [
            "192.168.137.231",
            "192.168.137.232",
            '192.168.137.230'
        ]
        device_list = find_cameras()
        mvcc_dev_info = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ip_addr = "%d.%d.%d.%d\n" % (((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24),
                                     ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16),
                                     ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8),
                                     (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff))
        if ip_addr == camera_target[0]:
            print(f"using: cam0{camera_id} {camera_target[camera_id]}")
            start_capture(device_list, camera_id, on_frame, to_exit=False)
        else:
            print(f"using: cam0{camera_id} {camera_target[camera_id]}")
            start_capture(device_list, 1 - camera_id, on_frame, to_exit=False)
        if not is_master:
            # B端，初始化完成，开始 on_frame，然后等待RPC调用
            rpc_server_url = f"http://{host_ips[camera_id]}:8000"
            print(f"rpc server will run on: {rpc_server_url}")
            start_rpc_server()
        else:
            rpc_display_url = f"http://{host_ips[2]}:8000"
            print(f"check if display started")
            # 如果C不启动则不连接终端
            # 所以需要先启动C终端
            global server_display
            try:
                server_display = xmlrpc.client.ServerProxy(rpc_display_url, allow_none=True)
                server_display.test()
            except Exception as e:
                print(f"{e}")
                server_display = None
            if server_display is None:
                print(f"No C display detected!")
            # 等待直到B节点启动完成
            rpc_server_url = f"http://{host_ips[1 - camera_id]}:8000"
            print(f"wait slave start...")
            slave_ok = False
            while not slave_ok:
                try:
                    server = xmlrpc.client.ServerProxy(rpc_server_url, allow_none=True)
                    server.test()
                    slave_ok = True
                except Exception as e:
                    print(f"{e}")
                    time.sleep(0.5)
            print(f"rpc server at: {rpc_server_url}")
            # 启动A(Master)进程
            th = threading.Thread(target=master_thread, daemon=True)
            th.start()
            while True:
                time.sleep(0.1)


if __name__ == '__main__':
    main()
