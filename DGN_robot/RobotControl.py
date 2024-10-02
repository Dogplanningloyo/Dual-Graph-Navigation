import os
import queue
import struct
import sys
import threading
import time

# import numpy as np
import keyboard
import serial

# import ARMS_MDH
from CRC import crc32_mpeg2

__version__ = 2.3


# 2.0 与无版本号与1.x版本区分
# 2.1 添加爪子物体被用户获取后松手的抓取函数
# 2.2 添加arm等待状态更新函数
# 2.3 添加不间断控制式

def reset():
    print('ready to restart program......')
    python = sys.executable
    os.execl(python, python, *sys.argv)


def BinToU8(data, site):
    data1 = data[site[0]:site[0] + 1]
    site[0] += 1
    return struct.unpack('<B', struct.pack('1B', *data1))[0]


def BinToS16(data, site):
    data1 = data[site[0]:site[0] + 2]
    site[0] += 2
    return struct.unpack('<h', struct.pack('2B', *data1))[0]


def BinToS32(data, site):
    data1 = data[site[0]:site[0] + 4]
    site[0] += 4
    return struct.unpack('<i', struct.pack('4B', *data1))[0]


def BinToFloat(data, site):
    data1 = data[site[0]:site[0] + 4]
    site[0] += 4
    return struct.unpack('<f', struct.pack('4B', *data1))[0]


def BinToDouble(data, site):
    data1 = data[site[0]:site[0] + 8]
    site[0] += 8
    return struct.unpack('<d', struct.pack('8B', *data1))[0]


def SendBufCalcCheck(data):
    data0 = []
    if type(data) == list:
        for data in data:
            data0 += BufCalcSum(data)
    else:
        data0 = BufCalcSum(data)
    return data0


def BufCalcSum(data):
    check_sum = 0
    for i in range(0, 10):
        check_sum += data[i]
    check_sum = check_sum.to_bytes(length=2, byteorder='big', signed=False)
    data += check_sum[1].to_bytes(length=1, byteorder='big', signed=False)
    return data


def BufCalc_CRC_Stm32(data, ContrastData=None):
    crcresult = struct.pack('I', crc32_mpeg2(data)).hex()
    if ContrastData == None:
        return crcresult
    else:
        Contrastresult = ContrastData.hex()
        if crcresult == Contrastresult:
            return True
        else:
            return False


class Chassis():
    def __init__(self, robotctrl):
        self.robotctrl = robotctrl

        class Set:
            def __init__(self):
                self.speedset = [0, 0, 0]
                self.location_setxyz = [0, 0, 0]
                self.reporttime = 2  # 数据回传间隔1-1000ms
                # 平移速度属性参数
                self.speedxy_max = 600  # 最大3000mm/s
                self.accelxy = 300  # mm/s2
                self.decelxy = 300
                # 旋转速度属性参数
                self.speedz_max = 12  # rpm
                self.accelz = 12  # rpm/s
                self.decelz = 12

        class State:
            def __init__(self):
                self.site_real = [0, 0, 0]  # 里程计位移 mm mm 圈 前正，左正，逆时针（向左转）正
                self.angle = [0, 0, 0]  # 右倾正，后倾正，向左转正
                self.screen_motor_angle = 0  # 显示屏角度 -179.9到+180

        self.set = Set()
        self.state = State()

    def setscreenangle(self, angle, IsAbsolute):  # 单位度 绝对angle范围-180到+180 相对angle右转正，左转负，0为显示器面对正右方
        angle *= 10  # 控制精度问题
        if not IsAbsolute:
            order_data = b'\x23' + b'\x06' + b'\x01' + angle.to_bytes(length=2, byteorder='big',
                                                                      signed=True) + b'\x00' + b'\x00' + b'\x00' + b'\x00' + b'\x00'

        else:

            order_data = b'\x23' + b'\x06' + b'\x00' + angle.to_bytes(length=2, byteorder='big',
                                                                      signed=True) + b'\x00' + b'\x00' + b'\x00' + b'\x00' + b'\x00'
        self.robotctrl.sendque.put(order_data)

    def setconf(self, reporttime, speedxy_max, accelxy_max, speedz_max, accelz_max):
        self.set.reporttime = reporttime  # 2ms 1-1000ms
        self.set.speedxy_max = speedxy_max  # 600  # 最大3000mm/s
        speedxy_max = int(speedxy_max / 20)
        self.set.accelxy = accelxy_max  # 300  # mm/s2
        self.set.decelxy = accelxy_max
        accelxy_max = int(accelxy_max / 10)
        self.set.speedz_max = speedz_max  # 12  # rpm
        self.set.accelz = accelz_max  # 12  # rpm/s
        self.set.decelz = accelz_max

        order_data = b'\x23' + b'\x05' + \
                     reporttime.to_bytes(length=2, byteorder='big', signed=False) + \
                     speedxy_max.to_bytes(length=1, byteorder='big', signed=False) + \
                     accelxy_max.to_bytes(length=1, byteorder='big', signed=False) + \
                     speedz_max.to_bytes(length=1, byteorder='big', signed=False) + \
                     accelz_max.to_bytes(length=1, byteorder='big', signed=False) + \
                     b'\x00' + b'\x00'
        self.robotctrl.sendque.put(order_data)

    def setspeed(self, speedset=None):
        if speedset is not None:
            self.set.speedset[0] = speedset[0]
            self.set.speedset[1] = speedset[1]
            self.set.speedset[2] = speedset[2]
            order_data = b'\x23' + b'\x01' + \
                         self.set.speedset[0].to_bytes(length=2, byteorder='big', signed=True) + \
                         self.set.speedset[1].to_bytes(length=2, byteorder='big', signed=True) + \
                         self.set.speedset[2].to_bytes(length=2, byteorder='big', signed=True) + \
                         b'\x01' + b'\x00'
            self.robotctrl.sendque.put(order_data)

    # 地盘运动 xy是以毫米为单位 z是30000分之1圈
    def setlocationXYZ(self, location=None):
        if location is not None:
            self.set.location_setxyz[0] = location[0]
            self.set.location_setxyz[1] = location[1]
            self.set.location_setxyz[2] = location[2]
            order_data = b'\x23' + b'\x07' + \
                         self.set.location_setxyz[0].to_bytes(length=2, byteorder='big', signed=True) + \
                         self.set.location_setxyz[1].to_bytes(length=2, byteorder='big', signed=True) + \
                         self.set.location_setxyz[2].to_bytes(length=2, byteorder='big', signed=True) + \
                         b'\x01' + b'\x00'
            self.robotctrl.sendque.put(order_data)


class Arms:
    def __init__(self, robotctrl):
        self.robotctrl = robotctrl

        class Set:
            def __init__(self):
                self.posset = []

        class State:
            def __init__(self):
                self.mdhangle = [0, 0, 0, 0, 0, 0, 0, 0]
                self.mdhpos = [0, 0, 0, 0, 0, 0]
                # 机械臂状态参数
                self.enable = [0, 0, 0, 0, 0, 0]
                self.reachposflag = [0, 0, 0, 0, 0, 0]
                self.error = [0, 0, 0, 0, 0, 0]
                self.posnow = [0, 0, 0, 0, 0, 0]
                # posnow面向电机输出面，中心顺时针转正
                # 1-机械臂末端向右正 2-机械臂末端向上正 3-机械臂末端向前负 4-机械臂末端向前正 5-机械爪水平90，顺直0 6-机械爪向右转负
                self.cs = 0
                self.angle_x = 0  # 前倾正
                self.angle_y = 0  # 右倾正
                self.camera_pos_real = {0, 0, 0, 0}  # 摄像头中心坐标相对于机器人最底部平面中心的{X,Y,r，俯仰角},故含机械臂整体的倾角

        self.set = Set()
        self.state = State()

    def setangle1(self, angle, IsAbsolute):
        if len(angle) != 6:
            return
        if not IsAbsolute:
            order_data = [b'\x23' + b'\x22' + struct.pack('<f', angle[0]) + struct.pack('<f', angle[1])
                , b'\x23' + b'\x23' + struct.pack('<f', angle[2]) + struct.pack('<f', angle[3])
                , b'\x23' + b'\x24' + struct.pack('<f', angle[4]) + struct.pack('<f', angle[5])]
        else:
            order_data = [b'\x23' + b'\x32' + struct.pack('<f', angle[0]) + struct.pack('<f', angle[1])
                , b'\x23' + b'\x33' + struct.pack('<f', angle[2]) + struct.pack('<f', angle[3])
                , b'\x23' + b'\x34' + struct.pack('<f', angle[4]) + struct.pack('<f', angle[5])]
        # TODO 可能存在bug，单次设置机械臂，可能因为can仲裁失败而设置失败
        for i in range(2):
            self.robotctrl.sendque.put(order_data)

    # 检查当前机械臂状态
    def check_reach(self):
        if sum(self.state.reachposflag) == 6:
            return True
        else:
            return False

    # 等待机械臂完成
    def wait_arm_over(self, wait_time=0.1):
        time.sleep(wait_time)  # 避免标志位未更新，导致直接跳出
        while not self.check_reach():  # 判断机械臂是否完成动作
            time.sleep(wait_time)


class Claw:
    def __init__(self, robotctrl):
        self.robotctrl = robotctrl

        class Set:
            def __init__(self):
                # 机械爪参数
                self.claw_speed = 0  # 0-18000,18000/540=33.33mm/s
                self.claw_force = 0  # 0-500N
                self.claw_pos = 0  # 0-150mm,完全张开为0
                self.AutoCalFlag = 1

        class State:
            def __init__(self):
                self.AutoCalFlag = 0
                self.MotorZeroFlag = 0
                self.pos = 0
                self.force = 0
                self.speed = 0
                self.MotorCurrent = 0

        self.set = Set()
        self.state = State()

    def setclaw(self, claw_pos, claw_speed, claw_force):
        CESET = 1
        findzero = 0
        self.set.claw_pos = claw_pos
        self.set.claw_speed = claw_speed
        self.set.claw_force = claw_force
        self.set.claw_pos = min(self.set.claw_pos, 255)
        self.set.claw_pos = max(self.set.claw_pos, 0)
        self.set.claw_speed = min(self.set.claw_speed, 18000)
        self.set.claw_speed = max(self.set.claw_speed, 0)
        self.set.claw_force = min(self.set.claw_force, 500)
        self.set.claw_force = max(self.set.claw_force, 0)
        order_data = b'\x23' + b'\x21' + struct.pack('<B', CESET) + struct.pack('<B', findzero) + struct.pack('<B',
                                                                                                              self.set.AutoCalFlag) + struct.pack(
            '<B', round(self.set.claw_pos)) + struct.pack('<H', round(self.set.claw_speed)) + struct.pack('<H', round(
            self.set.claw_force))
        for i in range(2):  # TODO can通信可能会造成抢断仲裁失败，导致数据发送失败，发送两次，通过降低性能来提高可靠性
            self.robotctrl.sendque.put(order_data)

    def findzero(self):
        CESET = 1
        findzero = 1
        order_data = b'\x23' + b'\x21' + struct.pack('<B', CESET) + struct.pack('<B', findzero) + struct.pack('<B',
                                                                                                              self.set.AutoCalFlag) + struct.pack(
            '<B', round(self.set.claw_pos)) + struct.pack('<H', round(self.set.claw_speed)) + struct.pack('<H', round(
            self.set.claw_force))
        self.robotctrl.sendque.put(order_data)

    # 爪子抓取后，感受到力变化后松爪
    def auto_grab_loosen(self):
        # self.setclaw(0, 18000, 20)  # 爪子开
        # time.sleep(2.5)
        # self.setclaw(150, 18000, 20)  # 爪子合
        # time.sleep(3.5)
        # warnings.warn("爪子维修，修复后应删去")
        # return False
        force_get_now = self.state.force
        time.sleep(0.1)
        while 1:
            # self.setclaw(0, 800, 20)  # 爪子开
            force_get = self.state.force
            speed_get = self.state.speed
            pose_get = self.state.pos
            print("The force is:", force_get)
            # print(f"The speed_get:{} pose_get:{}", speed_get, pose_get)
            time.sleep(0.01)
            if force_get > 40 or force_get > force_get_now + 4 or force_get < force_get_now - 4:
                # print("Open")
                self.setclaw(0, 18000, 20)  # 爪子合
                time.sleep(2.5)
                break


class RobotControl:
    def __init__(self, port='COM3', baudrate=2000000, queue_len=100, serialftpsPrint=True, threadinginside=False,
                 keyboardcontrol=False,
                 serialerroroutput=False):
        # self.ArmsMDH = ARMS_MDH.ArmsMDH_t()
        # 数据参数
        self.port = port
        self.baudrate = baudrate
        self.sendque = queue.Queue(queue_len)
        self.serialftpsPrint = serialftpsPrint
        self.serialerroroutput = serialerroroutput

        self.datalength = 1 + 4 * 7 + 8 * 3 + 8 * 9 + 4
        self.timeout = 0.1  # 1 / self.baudrate * 10 * (self.datalength + 2)
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout, parity='N')

        self.ftps_tx_cnt = 0
        self.ftps_tx_effective = 0
        self.ftps_rx_cnt = 0
        self.ftps_rx_correct = 0
        self.ftps_rx_cnt_all = 0
        self.ftps_rx_correct_all = 0
        self.ftps_rx_lastprinttime = 0
        self.dealdata_site = [0]

        self.chassis = Chassis(self)
        self.arms = Arms(self)
        self.claw = Claw(self)

        # 解析的下位机数据
        class State:
            def __init__(self):
                self.move_state = 0  # 0-失能 1-速度控制 3-X位置控制 5-Y位置控制 9-W位置控制 便于使用二进制的与操作
                self.battery_soc = 0  # 电池电量0-100
                self.temperature = 0  # 机器人控制芯片温度

        self.state = State()

        if threadinginside:
            t1 = threading.Thread(target=self.receivedata)
            t1.setDaemon(True)
            t1.start()
            t2 = threading.Thread(target=self.senddata)
            t2.setDaemon(True)
            t2.start()
            if keyboardcontrol:
                t3 = threading.Thread(target=self.keycontroldef)
                t3.setDaemon(True)
                t3.start()

    def check_move_over_in_pose(self):
        # 当前运动已结束
        if self.state.move_state == 1 or self.state.move_state == 0:
            return True
        elif self.state.move_state == 3 or 7 or 9:
            return False
        else:
            print("当前为其他模式，模式标志位为:", self.state.move_state)
            assert False, "模式错误，请检查，请联系lsy确定"

    def UpdateCaremaPos(self):
        camera_pos_base = 0

    def delay(self, s):
        s -= 0.0005
        time1 = time.time()
        while time.time() - time1 < s:
            continue

    def receivedata(self):
        while True:
            data = self.serial.read(self.datalength)
            if time.time() - self.ftps_rx_lastprinttime > 1:
                self.ftps_rx_cnt_all += self.ftps_rx_cnt
                self.ftps_rx_correct_all += self.ftps_rx_correct
                if self.serialftpsPrint:
                    if self.ftps_rx_cnt_all != 0:
                        ftps_rx_cnt_percent_all = (
                                                          self.ftps_rx_cnt_all - self.ftps_rx_correct_all) / self.ftps_rx_cnt_all * 100
                    else:
                        ftps_rx_cnt_percent_all = 0
                    # print("串口接收：%dHz/%d条，错误数据：%dHz/%d条，错误率%.2f%%，串口发送：%dHz，发送非心跳包：%dHz" % (
                    #     self.ftps_rx_cnt, self.ftps_rx_cnt_all, self.ftps_rx_cnt - self.ftps_rx_correct,
                    #     self.ftps_rx_cnt_all - self.ftps_rx_correct_all, ftps_rx_cnt_percent_all, self.ftps_tx_cnt,
                    #     self.ftps_tx_effective))
                self.ftps_rx_cnt = 0
                self.ftps_rx_correct = 0
                self.ftps_rx_lastprinttime = time.time()
                self.ftps_tx_cnt = 0
                self.ftps_tx_effective = 0
            if data != b'':
                self.ftps_rx_cnt += 1
                if not len(data) == self.datalength:
                    if self.serialerroroutput:
                        print("数据长度错误,实际为：", len(data), "应当为：", self.datalength)
                else:
                    if not data.startswith(b'\xaa\xbb\xcc\xdd'):
                        errorstart = data.find(b'\xAA\xBB\xCC\xDD')
                        if self.serialerroroutput:
                            print("数据起始错误,实际为：", data[:4], "应当为：aa bb cc dd，查询起始位置为：", errorstart)
                        if errorstart > 0:
                            self.serial.read(errorstart)
                        else:
                            self.serial.read(int(self.datalength / 2))
                    else:
                        if not BufCalc_CRC_Stm32(data[:-4], data[-4:]):
                            if self.serialerroroutput:
                                print("数据校验错误,实际为：", BufCalc_CRC_Stm32(data[:-4]), "应当为：", data[-4:].hex())
                        else:
                            self.ftps_rx_correct += 1
                            self.dealdata_site = [4]
                            self.state.move_state = BinToU8(data, self.dealdata_site)  # 获得当前运动状态
                            self.state.battery_soc = BinToFloat(data, self.dealdata_site)
                            self.state.temperature = BinToFloat(data, self.dealdata_site)
                            self.chassis.state.angle[0] = BinToFloat(data, self.dealdata_site)
                            self.chassis.state.angle[1] = BinToFloat(data, self.dealdata_site)
                            self.chassis.state.angle[2] = BinToFloat(data, self.dealdata_site)
                            self.chassis.state.screen_motor_angle = BinToS32(data, self.dealdata_site) / 8192 * 360
                            self.chassis.state.x_real = BinToDouble(data, self.dealdata_site)
                            self.chassis.state.y_real = BinToDouble(data, self.dealdata_site)
                            self.chassis.state.z_real = BinToDouble(data, self.dealdata_site)
                            for i in range(0, 6):
                                self.arms.state.enable[i] = BinToU8(data, self.dealdata_site)
                                self.arms.state.reachposflag[i] = BinToU8(data, self.dealdata_site)
                                self.arms.state.error[i] = BinToU8(data, self.dealdata_site)
                                self.arms.state.posnow[i] = BinToFloat(data, self.dealdata_site)
                                BinToU8(data, self.dealdata_site)
                                # print("arms.state.posnow:", self.arms.state.posnow)
                            # print(sum(self.arms.state.reachposflag))
                            self.claw.state.pos = BinToFloat(data, self.dealdata_site)
                            self.claw.state.force = BinToFloat(data, self.dealdata_site)
                            self.claw.state.speed = BinToS16(data, self.dealdata_site)
                            self.claw.state.MotorCurrent = BinToS16(data, self.dealdata_site)
                            self.claw.state.AutoCalFlag = BinToU8(data, self.dealdata_site)
                            self.claw.state.MotorZeroFlag = BinToU8(data, self.dealdata_site)
                            self.arms.state.cs = BinToU8(data, self.dealdata_site)
                            BinToU8(data, self.dealdata_site)
                            # print(self.state.Claw.__dict__.items())
                            self.arms.state.angle_x = BinToFloat(data, self.dealdata_site)
                            self.arms.state.angle_y = BinToFloat(data, self.dealdata_site)
                            # print('self.arms.state.angle_x.y',self.arms.state.angle_x,self.arms.state.angle_y)
                            self.arms.state.mdhangle = [self.arms.state.posnow[0], self.arms.state.angle_x,
                                                        self.arms.state.angle_y,
                                                        self.arms.state.posnow[1], self.arms.state.posnow[2],
                                                        self.arms.state.posnow[3], self.arms.state.posnow[4],
                                                        self.arms.state.posnow[5]]
                            # self.arms.state.mdhpos = self.ArmsMDH.PosSolve(self.arms.state.mdhangle)

    def senddata(self):

        while True:
            try:
                data = self.sendque.get_nowait()
                self.ftps_tx_effective += 1
                # print("当前发送队列剩余：", self.sendque.qsize()+1,data)
            except:
                data = b'\x23' + b'\x00' + b'\x00' + b'\x00' + b'\x00' + b'\x01' + b'\x00' + b'\x00' + b'\x00' + b'\x00'
                # print("当前发送队列剩余：0  ————发送心跳包")
                time.sleep(0.001)
            self.ftps_tx_cnt += 1
            self.serial.write(SendBufCalcCheck(data))

    def keycontroldef(self):
        while True:
            speedset = [0, 0, 0]
            if keyboard.is_pressed('W'):
                speedset[0] = self.chassis.set.speedxy_max * 10
            if keyboard.is_pressed('S'):
                speedset[0] = -self.chassis.set.speedxy_max * 10
            if keyboard.is_pressed('A'):
                speedset[1] = self.chassis.set.speedxy_max * 10
            if keyboard.is_pressed('D'):
                speedset[1] = -self.chassis.set.speedxy_max * 10
            if keyboard.is_pressed('Q'):
                speedset[2] = self.chassis.set.speedz_max * 100
            if keyboard.is_pressed('E'):
                speedset[2] = -self.chassis.set.speedz_max * 100
            self.chassis.setspeed(speedset)

            if keyboard.is_pressed('R'):  # 屏幕向右增加1度
                self.chassis.setscreenangle(1, False)
            if keyboard.is_pressed('T'):  # 屏幕向左增加1度
                self.chassis.setscreenangle(-1, False)

            if keyboard.is_pressed('J'):  # 机械臂底座左转
                self.arms.setangle1([-0.5, 0, 0, 0, 0, 0], False)
            if keyboard.is_pressed('L'):  # 机械臂底座右转
                self.arms.setangle1([0.5, 0, 0, 0, 0, 0], False)
            if keyboard.is_pressed('Y'):  # 机械臂主轴向上
                self.arms.setangle1([0, 1, 0, 0, 0, 0], False)
            if keyboard.is_pressed('H'):  # 机械臂主轴向下
                self.arms.setangle1([0, -1, 0, 0, 0, 0], False)
            if keyboard.is_pressed('I'):  # 机械臂关节向前
                self.arms.setangle1([0, 0, 1, 0, 0, 0], False)
            if keyboard.is_pressed('K'):  # 机械臂底座向后
                self.arms.setangle1([0, 0, -1, 0, 0, 0], False)
            if keyboard.is_pressed('U'):  # 机械臂关节向上
                self.arms.setangle1([0, 0, 0, 1, 0, 0], False)
            if keyboard.is_pressed('O'):  # 机械臂底座向下
                self.arms.setangle1([0, 0, 0, -1, 0, 0], False)
            if keyboard.is_pressed('P'):  # 机械臂手腕向上
                self.arms.setangle1([0, 0, 0, 0, 1, 0], False)
            if keyboard.is_pressed(';'):  # 机械臂手腕向下
                self.arms.setangle1([0, 0, 0, 0, -1, 0], False)
            if keyboard.is_pressed('['):  # 机械臂手腕右转
                self.arms.setangle1([0, 0, 0, 0, 0, 1], False)
            if keyboard.is_pressed(']'):  # 机械臂手腕左转
                self.arms.setangle1([0, 0, 0, 0, 0, -1], False)
            if keyboard.is_pressed('"'):  # 回原点
                self.arms.setangle1([0, 0, 0, 0, 0, 0], True)

            if keyboard.is_pressed(','):  # 爪子开
                self.claw.setclaw(self.claw.state.pos + 5, 18000, 50)
            if keyboard.is_pressed('.'):  # 爪子合
                self.claw.setclaw(self.claw.state.pos - 5, 18000, 50)

            time.sleep(0.01)

    # 导航控制函数 #FIXME 一定要停留3s？
    # def path_control(self, path):  # 路径控制函数
    #     # position = [0, 0, 0]
    #     for i in range(len(path) - 1):
    #         position = [0, 0, 0]
    #         x, y = int(path[i + 1][0] - path[i][0]), -int(path[i + 1][1] - path[i][1])
    #         position[0] = x * 10
    #         position[1] = y * 10
    #         self.chassis.setlocationXYZ(position)
    #         time.sleep(3.0)
    def path_control(self, path):  # 路径控制函数
        # position = [0, 0, 0]
        self.chassis.setspeed([0, 0, 0])
        self.chassis.setconf(2, 600, 200, 12, 4)
        for i in range(len(path) - 1):
            position = [0, 0, 0]
            x, y = int(path[i + 1][0] - path[i][0]), -int(path[i + 1][1] - path[i][1])
            position[0] = x * 10
            position[1] = y * 10
            self.chassis.setlocationXYZ(position)
            time.sleep(0.2)
            while not self.check_move_over_in_pose():
                pass


if __name__ == '__main__':
    robot_control = RobotControl(threadinginside=True, queue_len=10, keyboardcontrol=True)
    # robot_control.chassis.setspeed([0, 0, 0])
    # robot_control.chassis.setconf(4, 400, 400, 2, 4)
    # # robot_control.claw.setclaw(150, 18000, 500)# 爪子关
    # # time.sleep(2.5)
    # robot_control.claw.setclaw(0, 18000, 30)  # 爪子开
    # # robot_control.claw.setclaw(10, 180, 30)  # 爪子开
    # time.sleep(2.5)
    # robot_control.claw.setclaw(150, 18000, 30)# 爪子关
    # time.sleep(3.5)

    # # robot_control.arms.setangle1([0, 0, 0, 0, 0, 0], True)# 回原点
    # # robot_control.arms.setangle1([0, 0, 0, 0, 0, 0], True)# 回原点
    robot_control.chassis.setscreenangle(90, True)  # 屏幕向前
    # robot_control.chassis.setscreenangle(90, True)  # 屏幕向前

    # robot_control.claw.auto_grab_loosen()
    # time.sleep(3)
    # robot_control.arms.setangle1([0, 300, 300, 0, 0, 0], True)
    # robot_control.arms.setangle1([0, 300, 300, 0, 0, 0], True)
    print("机械臂运动")
    # robot_control.arms.setangle1([1.6785711960960477, 0, 945.8704362773594, 424.26683565036024, 0, 0], True)
    print("操作结束")
