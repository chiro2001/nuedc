# -- coding: utf-8 --

import time
import sys
import threading
import cv2
import numpy as np
import traceback

from ctypes import *

# sys.path.append("../MvImport")
from MvImport.MvCameraControl_class import *


# 为ProgressThread线程定义一个函数
def progress_thread(cam=0, nMode=0):
    stFileAccessProgress = MV_CC_FILE_ACCESS_PROGRESS()
    memset(byref(stFileAccessProgress), 0, sizeof(stFileAccessProgress))
    while True:
        # ch:获取文件存取进度 |en:Get progress of file access
        ret = cam.MV_CC_GetFileAccessProgress(stFileAccessProgress)
        print("State = [%x],Completed = [%d],Total = [%d]" % (
            ret, stFileAccessProgress.nCompleted, stFileAccessProgress.nTotal))
        if (ret != MV_OK or (
                stFileAccessProgress.nCompleted != 0 and stFileAccessProgress.nCompleted == stFileAccessProgress.nTotal)):
            print('press a key to continue.')
            break


# 为FileAccessThread线程定义一个函数
def file_access_thread(cam=0, nMode=0, filename: str = 'UserSet1', dev_filename: str = 'UserSet1'):
    if cam is None:
        return
    stFileAccess = f'{filename}.ini'
    # stFileAccess = MV_CC_FILE_ACCESS()
    # memset(byref(stFileAccess), 0, sizeof(stFileAccess))
    # # print(os.listdir('.'))
    # stFileAccess.pUserFileName = (f'{filename}.ini').encode('ascii')
    # stFileAccess.pDevFileName = dev_filename.encode('ascii')
    if 1 == nMode:
        # ch:读模式 |en:Read mode
        ret = cam.MV_CC_FeatureSave(stFileAccess)
        if MV_OK != ret:
            print("file access read fail ret [0x%x]\n" % ret)
    elif 2 == nMode:
        # ch:写模式 |en:Write mode
        try:
            ret = cam.MV_CC_FeatureLoad(stFileAccess)
            if MV_OK != ret:
                print("file access write fail ret [0x%x]\n" % ret)
                raise RuntimeError("Error Write")
        except Exception as e:
            traceback.print_exc()
            print(f"retry...")
            time.sleep(2)
            ret = cam.MV_CC_FeatureLoad(stFileAccess)
            if MV_OK != ret:
                print("file access write fail ret [0x%x]\n" % ret)
                raise RuntimeError("Error Write")


def update_config(cam, filename: str, on_pause):
    # while True:
    #     if g_pause is None:
    #         break
    #     else:
    #         time.sleep(0.1)
    # print(f"cam: {cam}")
    # on_pause(True)
    #
    # on_pause(False)

    on_pause(filename)


def test_it():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no Device!")
        sys.exit()

    print("find %d devices!" % deviceList.nDeviceNum)

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

    nConnectionNum = input("please input the number of the device to connect:")

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

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

    nPacketSize = cam.MV_CC_GetOptimalPacketSize()
    if int(nPacketSize) > 0:
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        if ret != 0:
            print("Warning: Set Packet Size fail! ret[0x%x]" % ret)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    data_buf = (c_ubyte * nPayloadSize)()
    pData = byref(data_buf)
    # for i in range(100):
    #     cam.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo, 1000)
    #     frame = np.frombuffer(bytes(pData._obj)[nPayloadSize - stFrameInfo.nWidth * stFrameInfo.nHeight:],
    #                           dtype=np.uint8).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    #     cv2.imshow("frame", frame)
    #     cv2.waitKey(1)

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    # # ch:读模式 |en:Read mode
    # print("read to file.")
    # print('press a key to start.')
    # # input("getch")
    # input("")
    #
    # try:
    #     hReadThreadHandle = threading.Thread(target=file_access_thread, args=(cam, 1, "small"))
    #     hReadThreadHandle.start()
    #     # time.sleep(0.005)
    #     # hProgress1ThreadHandle = threading.Thread(target=progress_thread, args=(cam, 1))
    #     # hProgress1ThreadHandle.start()
    # except:
    #     print("error: unable to start thread")
    #
    # print("waiting.")
    # # input("getch")
    # input("")
    #
    # hReadThreadHandle.join()
    # # hProgress1ThreadHandle.join()

    # ch:写模式 |en:Write mode
    print("write from file.")
    print('press a key to start.')
    # input("getch")
    # input("")

    try:
        hWriteThreadHandle = threading.Thread(target=file_access_thread, args=(cam, 2, 'big'))
        hWriteThreadHandle.start()
        # time.sleep(0.005)
        # hProgress2ThreadHandle = threading.Thread(target=progress_thread, args=(cam, 2))
        # hProgress2ThreadHandle.start()
    except:
        print("error: unable to start thread")

    print("waiting.")
    input("getch")

    hWriteThreadHandle.join()
    # hProgress2ThreadHandle.join()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()


if __name__ == '__main__':
    test_it()
