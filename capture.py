# -- coding: utf-8 --

import cv2
import time
import numpy as np
import sys
import threading
import traceback

from ctypes import *

from MvImport.MvCameraControl_class import *
from update_config import file_access_thread

g_bExit = False
g_pause = False

g_data_size = None


# 为线程定义一个函数
def work_thread(cam, pData, nDataSize, on_frame):
    global g_bExit, g_pause

    def on_quit():
        global g_bExit
        g_bExit = True

    def on_pause(filename, val=True):
        # global g_pause
        # g_pause = True
        ret = cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        # print("stopped")
        # time.sleep(1)
        print(f"change settings to: {filename}.ini")
        file_access_thread(cam, 2, filename)
        # time.sleep(1)
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()
        # print("started")

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        if g_pause:
            time.sleep(0.1)
        else:
            try:
                if g_pause:
                    # g_pause = None
                    pass
                else:
                    # time.sleep(0.1)
                    # on_frame(None, on_quit=on_quit, info=stFrameInfo, cam=cam, on_pause=on_pause)
                    # continue
                    # if pData is None:
                    pData = byref(data_buf)
                    nDataSize = g_data_size
                    ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
                    if ret == 0:
                        # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                        #     stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                        frame = np.frombuffer(bytes(pData._obj)[nDataSize - stFrameInfo.nWidth * stFrameInfo.nHeight:],
                                              dtype=np.uint8).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                        # print(f"caped")
                        # print(f"frame: {frame.shape}")
                        # cv2.imshow("frame", frame)
                        # cv2.waitKey(1)
                        # print(help(stFrameInfo))
                        # frame = stFrameInfo.data

                        on_frame(frame, on_quit=on_quit, info=stFrameInfo, cam=cam, on_pause=on_pause)
                    else:
                        print("no data[0x%x]" % ret)
                        # cv2.destroyWindow("frame")
                    if g_bExit:
                        break
            except Exception as e:
                traceback.print_exc()


# cam = None
data_buf = None


def update_buf(cam):
    global data_buf, g_data_size
    if data_buf is not None:
        del data_buf
        data_buf = None
    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue
    g_data_size = nPayloadSize
    data_buf = (c_ubyte * nPayloadSize)()


def start_capture(deviceList, nConnectionNum, on_frame):
    # global cam
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    update_buf(cam)

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, None, g_data_size, on_frame))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    print("press a key to stop grabbing.")
    # msvcrt.getch()
    # input("Enter...")

    # g_bExit = True
    # hThreadHandle.join()
    while True:
        if g_bExit:
            hThreadHandle.join()
            break
        time.sleep(0.1)

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    del data_buf


def find_cameras():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)

    # nConnectionNum = input("please input the number of the device to connect:")
    # nConnectionNum = 0
    #
    # if int(nConnectionNum) >= deviceList.nDeviceNum:
    #     print("intput error!")
    #     sys.exit()
    return deviceList


if __name__ == "__main__":
    device_list = find_cameras()
    start_capture(device_list, 0)
