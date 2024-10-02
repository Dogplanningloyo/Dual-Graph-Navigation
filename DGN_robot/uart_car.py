# -*- coding: utf-8 -*-
# @Time    :2024/1/27 10:07
# @Author  :LSY Dreampoet
# @SoftWare:PyCharm
import time
import warnings
import serial
import numpy as np


# 地盘控制
class Uart_control:
    def __init__(self, port_move="COM14", port_camera="COM3", baudrate=115200, bytesize=8, stopbits=1, timeout=0.5):
        # 串口初始化
        # 底盘运动串口
        self.ser_move = serial.Serial()
        self.ser_move.port = port_move  # 端口号
        self.ser_move.baudrate = baudrate  # 波特率
        self.ser_move.bytesize = bytesize  # 数据位
        self.ser_move.stopbits = stopbits  # 停止位
        self.ser_move.timeout = timeout  # 超时时间
        try:
            self.ser_move.open()  # 打开串口,要找到对的串口号才会成功
        except Exception as e:
            warnings.warn("地盘控制串口打开失败！" + str(e))
        # 相机控制串口
        self.ser_camera = serial.Serial()
        self.ser_camera.port = port_camera  # 端口号
        self.ser_camera.baudrate = baudrate  # 波特率
        self.ser_camera.bytesize = bytesize  # 数据位
        self.ser_camera.stopbits = stopbits  # 停止位
        self.ser_camera.timeout = timeout  # 超时时间
        try:
            self.ser_camera.open()  # 打开串口,要找到对的串口号才会成功
        except Exception as e:
            warnings.warn("相机控制串口打开失败！" + str(e))

    # 校验和
    def sum_check(self, data_get):
        check_num = 0
        for check_i in data_get:
            check_num += check_i
        check_num = check_num.to_bytes(2, 'big', signed=False)
        check_num = check_num[-1].to_bytes(1, 'big', signed=False)
        return check_num

    # BCC校验位
    def bcc_check(self, data_get):
        bcc_num = 0
        for bcc_i in data_get:
            bcc_num ^= bcc_i
        bcc_num = bcc_num.to_bytes(1, 'big', signed=False)
        return bcc_num

    # 串口数据处理并发送
    def move_control(self, data_get):
        # 数据获取
        # print("Get data:", data_get)
        x_data = data_get[0].to_bytes(2, 'big', signed=True)
        y_data = data_get[1].to_bytes(2, 'big', signed=True)
        z_data = data_get[2].to_bytes(2, 'big', signed=True)
        # 帧头是0x7B，帧尾是x7D，一共11个字节
        data_get = b'\x7B' + b'\x00' + b'\x00' + x_data + y_data + z_data
        # 数据校验
        data_check_BCC = self.bcc_check(data_get)
        # 通过十六进制的格式输出数据

        data_send_all = data_get + data_check_BCC + b'\x7D'
        self.ser_move.write(data_send_all)  # 发送数据

    # 按照距离来确定速度
    def positional_control(self, xyz_get, speed=200, turn_speed=1000, acculation=1130):
        # 判断数值正负
        def judge_num(num):
            if num > 0:
                return 1
            elif num < 0:
                return -1
            else:
                return 0

        # x y 是二维平面坐标， z是旋转
        x_data = xyz_get[0]
        y_data = xyz_get[1]
        z_data = xyz_get[2]
        # 计算x和y构成的物理距离
        distance = (x_data ** 2 + y_data ** 2) ** 0.5
        # 计算加速时间
        v_a_t = speed / acculation
        # 计算剩余时间
        v_s_t = (distance - speed ** 2 / acculation) / speed
        # 计算总时间
        move_time = v_a_t * 2 + v_s_t
        print("move_time:", move_time)
        # move_time = distance / speed
        if x_data and y_data:
            # 判断x和y的正负
            move_set = [judge_num(x_data) * speed, judge_num(y_data) * speed, 0]
            # move_set = [speed, speed, 0]
            self.move_control(move_set)
            time.sleep(move_time)
            self.move_control([0, 0, 0])
        elif x_data:
            # 判断x的正负
            move_set = [judge_num(x_data) * speed, 0, 0]
            print("move_set:", move_set)
            self.move_control(move_set)
            time.sleep(move_time)
            self.move_control([0, 0, 0])
        elif y_data:
            # 判断y的正负
            move_set = [0, judge_num(y_data) * speed, 0]
            self.move_control(move_set)
            time.sleep(move_time)
            self.move_control([0, 0, 0])
        elif z_data:
            turn_time = (abs(z_data) * 1000) / (turn_speed * 360)
            move_set = [0, 0, 1000]
            self.move_control(move_set)
            time.sleep(turn_time)
            self.move_control([0, 0, 0])
        # time.sleep(0.1)
        # uart_send(data_get)

    # 底盘旋转
    def move_turn(self, angle):
        '''
        底盘旋转
        :param angle: 0:向左 1:向右
        :return:
        '''
        if angle == 0: #　向左
            self.move_control([00, 0, int(1600)])  # 0.001rad/s
            time.sleep(1)
            self.move_control([0, 0, 0])
        elif angle == 1: # 向右
            self.move_control([00, 0, int(-1600)])  # 0.001rad/s
            time.sleep(1)
            self.move_control([0, 0, 0])
        else:
            pass

    # 串口接收函数
    def move_receive(self):
        # 读取24字节数据
        data = self.ser_move.read(24)

        # 解析数据
        frame_header = data[0]
        disable_flag = data[1]
        x_speed = int.from_bytes(data[2:4], byteorder='big', signed=True)
        y_speed = int.from_bytes(data[4:6], byteorder='big', signed=True)
        z_speed = int.from_bytes(data[6:8], byteorder='big', signed=True)
        x_acceleration = int.from_bytes(data[8:10], byteorder='big', signed=True)
        y_acceleration = int.from_bytes(data[10:12], byteorder='big', signed=True)
        z_acceleration = int.from_bytes(data[12:14], byteorder='big', signed=True)
        x_angular_velocity = int.from_bytes(data[14:16], byteorder='big', signed=True)
        y_angular_velocity = int.from_bytes(data[16:18], byteorder='big', signed=True)
        z_angular_velocity = int.from_bytes(data[18:20], byteorder='big', signed=True)
        power_voltage = int.from_bytes(data[20:22], byteorder='big', signed=True)

        # 数据处理
        '''
        速度单位是mm/s，加速度单位是m/s^2，角速度单位是°/s，电压单位是V
        '''
        x_acceleration = x_acceleration * 19.6 / 32768
        y_acceleration = y_acceleration * 19.6 / 32768
        z_acceleration = z_acceleration * 19.6 / 32768
        x_angular_velocity = x_angular_velocity * 500 / 32768
        y_angular_velocity = y_angular_velocity * 500 / 32768
        z_angular_velocity = z_angular_velocity * 500 / 32768

        power_voltage /= 1000
        checksum = data[22]
        frame_tail = data[23]

        # 校验数据
        receive_check = self.bcc_check(data[0:22])
        if receive_check != checksum or frame_header != 0x7B:
            warnings.warn("数据接收头部或校验错误！")
        if disable_flag == 0x01:
            warnings.warn("设备失能！电池电压低于 10V，此时电池电量即将耗尽；使能开关打到了 OFF 端；此时处于开机 10 秒前的时间内")
        print("receive_check:", receive_check.hex())
        # 打印解析结果
        print("帧头：", hex(frame_header))
        print("失能标志位：", hex(disable_flag))
        print("X轴实时速度：", x_speed)
        print("Y轴实时速度：", y_speed)
        print("Z轴实时速度：", z_speed)
        print("X轴加速度：", x_acceleration)
        print("Y轴加速度：", y_acceleration)
        print("Z轴加速度：", z_acceleration)
        print("绕X轴的角速度：", x_angular_velocity)
        print("绕Y轴的角速度：", y_angular_velocity)
        print("绕Z轴的角速度：", z_angular_velocity)
        print("电源电压：", power_voltage)
        print("校验位：", hex(checksum))
        print("帧尾：", hex(frame_tail))

    # 相机转向控制
    def camera_control(self, angle_get):
        # 帧头为0xF1，校验为和校验
        data_ori = b'\xF1' + angle_get.to_bytes(1, 'big', signed=False)
        send_check = self.sum_check(data_ori)
        data_send = data_ori + send_check
        self.ser_camera.write(data_send)


if __name__ == '__main__':
    # ser = serial.Serial("COM14", 115200, timeout=0.5)  # 连接串口
    Robot_Control_uart = Uart_control(port_move="COM14", port_camera="COM6")
    # Robot_Control_uart.camera_control(90)
    # Robot_Control_uart.move_control([0, 0, int(np.pi*1000*2)])
    # Robot_Control_uart.move_control([-100, 0, 0])
    # Robot_Control_uart.positional_control([-100,0,0])
    # Robot_Control_uart.move_control([00, 0, int(np.pi*1000*2)])
    Robot_Control_uart.move_control([00, 0, int(1600)]) #　0.001rad/s
    time.sleep(1)
    Robot_Control_uart.move_control([0, 0, 0])

    # while 1:
    #     Robot_Control_uart.camera_control(45)
    #     time.sleep(0.1)
    # x_data = 276
    # y_data = 000
    # z_data = 0000
    # x_data = x_data.to_bytes(2, 'big', signed=True)
    # y_data = y_data.to_bytes(2, 'big', signed=False)
    # z_data = z_data.to_bytes(2, 'big', signed=False)
    # print("x_data:", x_data)
    # # 帧头是0x7B，帧尾是x7D，一共11个字节
    # data_get = b'\x7B' + b'\x00' + b'\x00' + x_data + y_data + z_data
    # # 数据校验
    # data_check_BCC = bcc_check(data_get)
    # # 通过十六进制的格式输出数据
    # print("Get data_get:", int.from_bytes(data_check_BCC, byteorder='big', signed=False))
    # # print("{:#04x}".format(110))
    # print("Get data_check_BCC:", data_check_BCC.hex())
    #
    #
    #
    # data_send_all = data_get + data_check_BCC + b'\x7D'
    # print("Get data_send_all:", data_send_all)
    # ser = serial.Serial("/COM1", 115200, timeout=0.5)  # 连接串口
    # ser.flushInput()  # 清空缓冲区
    # ser.flushOutput()
    # while True:
    #     try:
    #         count = ser.inWaiting()  # 获取串口缓冲区数据
    #         if count != 0:
    #             data = ser.read(count)  # 读取内容并显示
    #             print(data)
    #             ser.write(data)  # 将读取到的内容写入到串口
    #     except KeyboardInterrupt:
    #         if ser != None:
    #             ser.close()  # 关闭串口
    #         break

    # uart_send([-150, 0, 0])
    # time.sleep(1)
    # uart_send([0, 0, 0])
    # while True:
    #     uart_receive()
    #     time.sleep(1)
    # positional_control([420, 0, 0])
