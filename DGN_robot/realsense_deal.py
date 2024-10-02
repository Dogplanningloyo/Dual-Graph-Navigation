# -*- coding: utf-8 -*-
# @Time    :2021/11/14 0:09
# @Author  :LSY Dreampoet
# @SoftWare:PyCharm

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import random
import math
import scipy.linalg as linalg
import warnings
import open3d as o3d  # 点云操作
# 可视化函库 非必要
import matplotlib.pyplot as plt

# 运算加速
# from numba import jit

__version__ = 2.8


# TODO 更新日志
# 2.3 加入数学推导出相机位姿矫正的代码
# 2.4 版本更新,优化了坐标转化函数,提供了新的转化方式,并完成离散点的降采样和过滤 #增加依赖 open3d
# 2.41 版本维护，同步了保存与读取函数
# 2.5 版本添加了open3d可视化实例场景图
# 2.51 修订了原版本中rs_get_xyz函数坐标轴取反的问题
# 2.6 新增深度图可视化函数
# 2.7 角度取值顺序问题
# 2.8 修正了相机内参传入与使用问题

# RealScecse初始化函数， h为捕获图像高 w为宽 f为帧率 camera 类型 l515
def RealScecse_init(cameral_serial, w=1280, h=800, f=30, camera="D455", mode=1):
    '''
    :param cameral_serial: 相机id，可通过realsense sdk获取
    :param w: 图像宽度
    :param h: 图像高度
    :param f: 图像帧率
    :param camera: 相机型号
    :param mode: 选择相机启动模式
    :return: rgb视频流管道，depth视频流管道，配置文件
    '''
    # Init RealSense
    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config
    config.enable_device(cameral_serial)
    d_w = 1280
    d_h = 720
    # 自定义模式 高分辨率慢帧率，低分辨率高帧率，设置模式
    if mode == 1:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
    elif mode == 2:
        config.enable_stream(rs.stream.depth, d_w, d_h, rs.format.z16, f)  # 配置depth流
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)  # 配置color流
    profile = pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    # 使用 L515 可以调整距离，但是D455不可以使用以下代码调试
    print("尝试设置最小距离")
    # 距离单位是mm
    sensor_dep = profile.get_device().first_depth_sensor()
    if camera == "l515":
        dist = sensor_dep.get_option(rs.option.min_distance)
        print("目前最小的距离是：", dist)
        min_dist = 10
        print("开始设置最小距离：", min_dist)
        dist = sensor_dep.set_option(rs.option.min_distance, min_dist)
        dist = sensor_dep.get_option(rs.option.min_distance)
        print("新的最小距离为：", dist)
    return pipeline, align, profile


# 两个通道同时反馈rgbd与imu数据 注意！这是两个通道！！！非单通道
def RealScecse_init_with_imu(cameral_serial, imu_get=True):
    # 将管道分别设为图像和imu两个
    # rgbd传输管道
    rgbd_pipeline = rs.pipeline()
    rgbd_config = rs.config()
    rgbd_config.enable_device(cameral_serial)
    rgbd_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # depth
    rgbd_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # rgb
    rgbd_profile = rgbd_pipeline.start(rgbd_config)
    align_to_color = rs.stream.color
    align_color = rs.align(align_to_color)
    # 判断是否启用IMU数据
    if imu_get:
        # imu传输管道
        imu_pipeline = rs.pipeline()
        imu_config = rs.config()
        imu_config.enable_device(cameral_serial)
        imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # 250 200 63
        imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
        imu_profile = imu_pipeline.start(imu_config)
        align_to_acc = rs.stream.accel
        align_acc = rs.align(align_to_acc)
        return rgbd_pipeline, align_color, imu_pipeline, align_acc
    return rgbd_pipeline, align_color, None, None


# 单个通道同时反馈rgbd与imu数据
def RealScecse_init_with_imu_single(cameral_serial, select_custom=False, imu_num=200):
    '''
    :param cameral_serial:  深度相机序列号，可通过官方软件的inof中查询获得
    :param select_custom:(bool)  选择自定义分辨率,高清建图或特殊需求可用
    :return:图像通道，对齐的句柄
    '''
    # 将管道分别设为图像和imu两个
    # rgbd传输管道
    rgbd_imu_pipeline = rs.pipeline()
    rgbd_imu_config = rs.config()
    # rgbd_imu_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # rgbd_imu_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # rgbd_imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    # rgbd_imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
    if select_custom:
        rgbd_imu_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # depth
        rgbd_imu_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # rgb
    else:
        rgbd_imu_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 90)  # depth
        rgbd_imu_config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 90)  # rgb

    rgbd_imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, imu_num)  # 250 200 63
    rgbd_imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)  # 400 200

    rgbd_imu_config.enable_device(cameral_serial)
    rgbd_imu_pipeline.start(rgbd_imu_config)
    # rgbd_imu_profile = rgbd_imu_pipeline.start(rgbd_imu_config)
    align_to_color = rs.stream.color
    align_color = rs.align(align_to_color)
    return rgbd_imu_pipeline, align_color


# rgbd 初始化 gpt
def RealScecse_init_with_GPT(cam_serial):
    # 配置深度、RGB和IMU流
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
    config.enable_device(cam_serial)
    # 启动相机
    pipeline.start(config)

    align_to_color = rs.stream.color
    align_color = rs.align(align_to_color)
    return pipeline, align_color


# 返回完整数据 该函数不建议在实际工程中应用，仅作为测试
def get_aligned_images(pipeline, align, simple_or=0):
    '''
    :param pipeline: 视频流通道
    :param align: 对齐后的通道流
    :param simple_or: 标志位
    :return:相机内参、深度参数、彩色图、深度图、齐帧中的depth帧 3通道深度图
    '''
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    # camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
    #                      'ppx': intr.ppx, 'ppy': intr.ppy,
    #                      'height': intr.height, 'width': intr.width,
    #                      'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
    #                      }
    # 保存内参到本地
    # with open('../intrinsics.json', 'w') as fp:
    #     json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图
    # print("depth_image_8bit:", depth_image_8bit)

    # 伪彩色图
    depth_colormap_aligned = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
    # cv2.imshow("depth_colormap_aligned:",depth_colormap_aligned)
    # 返回pitch roll 绝对角度 yaw无法通过单次imu获得，默认为None
    accel = aligned_frames[2].as_motion_frame().get_motion_data()  # 获取加速度的数据
    # 需要注意 深度相机imu的xyz 和常规的右手坐标系并不一致
    # roll 复合角
    roll_yz = math.sqrt(accel.y ** 2 + accel.z ** 2)
    roll = np.arctan2(-accel.x, roll_yz)
    # pich 轴复合角
    # 重力加速度在X-Y面上的投影
    pitch_xy = math.sqrt(accel.x * accel.x + accel.y * accel.y)
    # 重力加速度在Z轴上的分量与在X-Y面上的投影的正切，即俯仰角
    pitch = np.arctan2(-accel.z, pitch_xy)  # 57.3 = 180/3.1415
    # row无法通过imu获得，需要图像等
    yaw = None
    angles = [yaw, pitch, roll]
    # 如果简化发送，返回相机彩色图
    if simple_or:
        return color_image
    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧 3通道深度图
    else:
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d, depth_colormap_aligned, angles


# 通过两通道分别获取rgbd + imu 可获取角度 注意是双通道！！不是单通道！别用错了
def get_aligned_rgbd_imu(rgbd_pipeline, align_color, imu_pipeline, align_acc):
    '''

    :param rgbd_pipeline: 图像流通道
    :param align_color: 图像对齐句柄
    :param imu_pipeline: IMU流通道
    :param align_acc: IMU对齐句柄
    :return:rgb图像，深度数据（不是图像），深度相机内参，IMU数据
    '''
    # 获取关键帧，得到rgbd数据
    rgbd_frames = rgbd_pipeline.wait_for_frames()
    aligned_rgbd_frames = align_color.process(rgbd_frames)
    # 图像获取
    color_frame = aligned_rgbd_frames.get_color_frame()
    depth_frame = aligned_rgbd_frames.get_depth_frame()  # 不要轻易将深度数据转化为numpy类型，在当前SDK下，这样只会保存原始数据，将会丢失大量其他的参数（2023-3-22）
    # 图像数据转换
    color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data()
    # 相机内参
    # rgbd_frame = rgbd_pipeline.wait_for_frames() # 读取相机内参将会占用15ms的时间
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

    # 获取关键帧，获得imu数据
    imu_frames = imu_pipeline.wait_for_frames()
    aligned_imu_frames = align_acc.process(imu_frames)
    accel = aligned_imu_frames[0].as_motion_frame().get_motion_data()  # 获取加速度的数据
    # 需要注意 深度相机imu的xyz 和常规的右手坐标系并不一致
    # roll 复合角
    roll_yz = math.sqrt(accel.y ** 2 + accel.z ** 2)
    roll = np.arctan2(-accel.x, roll_yz)

    # pich 轴复合角
    # 重力加速度在X-Y面上的投影
    pitch_xy = math.sqrt(accel.x * accel.x + accel.y * accel.y)
    # 重力加速度在Z轴上的分量与在X-Y面上的投影的正切，即俯仰角
    pitch = np.arctan2(-accel.z, pitch_xy)  # 57.3 = 180/3.1415

    # row无法通过imu获得，需要图像等
    yaw = None
    warnings.warn("该方法的yaw pitch roll角度绝对值准确,但是正负号和顺序可能有误,需要注意")
    angles = [yaw, roll, -pitch]

    return color_image, depth_frame, depth_intrin, angles


# 通过单通道直接获取rgbd + imu数据
def get_aligned_rgbd_imu_single(pipeline, align_color, fill_holl=True, show=False, np_change=True):
    '''

    :param pipeline:数据流通道
    :param align_color:数据对齐句柄
    :param fill_holl:(bool) 选择是否启动空洞填补
    :param show:(bool) 选择是否可视化填补效果
    :return: rgb图像，深度数据（不是图像），深度相机内参，IMU数据
    '''
    # 获取关键帧，得到rgbd数据
    rgbd_frames = pipeline.wait_for_frames()
    aligned_rgbd_frames = align_color.process(rgbd_frames)
    # 图像获取
    color_frame = aligned_rgbd_frames.get_color_frame()
    depth_frame = aligned_rgbd_frames.get_depth_frame()

    # 图像数据转换
    color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data()) # 不要轻易将深度数据转化为numpy类型，在当前SDK下，这样只会保存原始数据，将会丢失大量其他的参数（2023-3-22）

    # 相机内参
    # rgbd_frame = rgbd_pipeline.wait_for_frames() # 读取相机内参将会占用15ms的时间
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

    # 获取IMU数据    # 需要注意 深度相机imu的xyz 和常规的右手坐标系并不一致 具体可打开realsense软件观测
    accel = aligned_rgbd_frames[2].as_motion_frame().get_motion_data()  # 获取加速度的数据
    # roll 复合角
    roll_yz = math.sqrt(accel.y ** 2 + accel.z ** 2)
    roll = np.arctan(accel.x / roll_yz)
    # pich 轴复合角
    # 重力加速度在X-Y面上的投影
    pitch_xy = math.sqrt(accel.x * accel.x + accel.y * accel.y)
    # 重力加速度在Z轴上的分量与在X-Y面上的投影的正切，即俯仰角
    pitch = np.arctan(accel.z / pitch_xy)  # 57.3 = 180/3.1415
    # yaw无法通过imu获得，需要图像、积分等方法求得
    yaw = 0
    # FIXME 角度选择？
    # angles = [yaw, roll, -pitch]  # pitch角度需要考虑相位判断
    angles = [yaw, roll, pitch]  # pitch角度需要考虑相位判断
    if fill_holl:
        # 空间滤波器是域转换边缘保留平滑的快速实现
        spatial = rs.spatial_filter()

        # 我们可以通过增加smooth_alpha和smooth_delta选项来强调滤镜的效果：
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        # The filter also offers some basic spatial hole filling capabilities:
        # 该过滤器还提供一些基本的空间孔填充功能：
        spatial.set_option(rs.option.holes_fill, 3)  # 可调整大小,选择填补的块大小
        filtered_depth = spatial.process(depth_frame)
        # show = False # 选择是否可视化填补效果
        if show:
            colorizer = rs.colorizer()  # 着色器 测试可观测空洞填补效果
            depth_frame = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
            cv2.imshow("original data", depth_frame)
            cv2.imshow("colorized_depth", colorized_depth)
            cv2.waitKey(1)
        if np_change:  # 强制转化为np类型，可进行后续更改
            filtered_depth = np.asarray(filtered_depth.get_data(), np.uint16)
        return color_image, filtered_depth, depth_intrin, angles

    if np_change:  # 强制转化为np类型，可进行后续更改
        depth_frame = np.asarray(depth_frame.get_data(), np.uint16)
    return color_image, depth_frame, depth_intrin, angles


# 截取图像
def crop_image(img, a, b, c, d):
    '''
    :param img: 传入需要图像
    :param a: 起始x
    :param b: 终止x
    :param c: 起始y
    :param d: 终止y
    :return: 剪切后的图像
    '''
    cropImg = img[a:b, c:d]  # 裁剪图像
    return cropImg


# 返回像素点对应的三维坐标，change = ture转换为当前xyz坐标系
def rs_get_xyz(aligned_depth_frame, depth_intrin, center):
    '''
    :param aligned_depth_frame:对齐好的深度图像数据或者对应numpy数据
    :param depth_intrin:深度相机内参
    :param center:目标点像素坐标
    :return:真实空间中的xyz 单位mm 不是米！！是毫米！ x左 y 向下 z向前
    '''
    # realsense 官方的函数
    center_get = [int(center[1]), int(center[0])]  # 注意官方的和实际的 像素坐标 xy是相反的
    # center_get = [int(center[0]), int(center[1])]  # 此bug已修复？
    # 判断像素点距离来源
    if type(aligned_depth_frame).__name__ == "depth_frame":
        dis = aligned_depth_frame.get_distance(center_get[1], center_get[0]) * 1000  # (x, y)点的真实深度值 # 要保持单位统一，转化为mm
    elif type(aligned_depth_frame) == np.ndarray:
        dis = aligned_depth_frame[center_get[0]][center_get[1]]
        # 可视化深度图
        # plt.imshow(aligned_depth_frame)
        # plt.show()
        # print("dis:", dis)
    elif type(aligned_depth_frame).__name__ == "frame":
        depth_change = np.asanyarray(aligned_depth_frame.get_data())
        dis = depth_change[center_get[0]][center_get[1]]
    else:
        dis = None
        print("当前数据类型为：", type(aligned_depth_frame).__name__)
        assert dis != None, "输入深度图像数据类型错误，需要为pyrealsense2.frame Z16 #0或numpy.ndarray类型，当前输入的类型为"
    # print("dis:", dis)
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                        [center[0], center[1]],  # 此处使用正常xyz（来自董炳成的版本检查）
                                                        dis)

    return camera_coordinate


# 通过内参与深度值，直接转化坐标消息

# 输入图像像素坐标,转换为实际空间坐标
def pixel2xyz_get(depth_image_np, get_cameraInfo, center, rgb=False):
    # warnings.warn("此函数已被禁用，由 rs_get_xyz 函数替换", DeprecationWarning)
    # assert rgb, "此函数已被禁用，由 rs_get_xyz 函数替换"
    # while 1:
    try:
        depth_z = depth_image_np[center[0]][center[1]]  # 获取的深度数据 # 与对应xy是相反的
        # 当遇到深度“坏点”（深度数据为0）的数据时，检测周围数据，避开坏点
        # [dy, dx] = center  # Attention y x instead of x y
        # depth_z = depth_image_np[center[0]][center[1]]  # 获取的深度数据 # 与对应xy是相反的
        [dx, dy] = center  # Attention y x instead of x y

        while depth_z == 0:
            dy += 5 + random.randint(-10, 10)
            dx += 5 + random.randint(-10, 10)
            depth_z = depth_image_np[dx][dy]
            print("像素点深度为0，重新获取深度数据")
        # 深度数据转像素坐标
        # CAM_WID, CAM_HGT = get_cameraInfo['width'], get_cameraInfo['height']
        # CAM_FX, CAM_FY = get_cameraInfo['fx'], get_cameraInfo['fy']
        # CAM_CX, CAM_CY = get_cameraInfo['ppx'], get_cameraInfo['ppy']
        # 内参矩阵
        [[CAM_FX, z0, CAM_CX],
             [z0, CAM_FY, CAM_CY],
             [z0, z0, z1]] = get_cameraInfo.intrinsic_matrix
        # 畸变函数
        # K = get_cameraInfo['coeffs']
        # 转换
        x_pixel = int(dx - CAM_CX)
        y_pixel = int(dy - CAM_CY)
        if False:  # 如果需要矫正视线到Z的转换的话使能
            f = (CAM_FX + CAM_FY) / 2.0
            img_z *= f / np.sqrt(x ** 2 + y ** 2 + f ** 2)
        depth_x = depth_z * x_pixel / CAM_FX  # X=Z*(u-cx)/fx
        depth_y = depth_z * y_pixel / CAM_FY  # Y=Z*(v-cy)/fy
        pc = np.array([depth_x, depth_y, depth_z])  # 返回顺序为: x y z 右手坐标系
        # print("pc:", pc)
        return pc
    except TypeError as e:
        print("没有获取到数据,得到None", e)
    except IndexError as e:
        print("超出范围后，仍然没有找到合适的深度数据，直接返回 0 0 0 ", e)
        return np.array([0, 0, 0])


# Save data
def rs_data_save(rgb, aligned_depth_frame, depth_intrin, angels, output_path="./realsense_data/"):
    '''

    :param rgb: 图像数据
    :param aligned_depth_frame: 对齐后的深度数据
    :param depth_intrin:深度相机内参
    :param angels:拍摄时的位姿
    :param output_path:存储位置
    :return:None
    '''
    # image 文件夹
    # depth 文件夹
    # 内参 文件夹
    img_dir = output_path + 'image/'
    dep_dir = output_path + 'depth/'
    int_dir = output_path + 'intrin/'
    # 文件夹是否存在判断
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(dep_dir):
        os.makedirs(dep_dir)
    if not os.path.exists(int_dir):
        os.makedirs(int_dir)
    # 判断当前文件数量
    file_num = len(os.listdir(img_dir))

    # 保存深度图像数据和彩色图像数据
    if type(aligned_depth_frame).__name__ == "depth_frame":
        aligned_depth_frame_np = aligned_depth_frame.get_data()
    elif type(aligned_depth_frame) == np.ndarray:
        aligned_depth_frame_np = aligned_depth_frame
    elif type(aligned_depth_frame).__name__ == "frame":
        aligned_depth_frame_np = aligned_depth_frame.get_data()
    else:
        aligned_depth_frame_np = None
        print("当前数据类型为：", type(aligned_depth_frame_np).__name__)
        assert aligned_depth_frame_np != None, "输入深度图像数据类型错误，需要为pyrealsense2.frame Z16 #0或numpy.ndarray类型，当前输入的类型为"
    depth_image_np = np.asarray(aligned_depth_frame_np, np.uint16)
    cv2.imwrite(img_dir + str(file_num) + 'rgb_img.png', rgb)
    np.save(dep_dir + str(file_num) + 'depth_np.npy', depth_image_np)
    # 保存相机内参 与 imu数据
    np.savez(int_dir + str(file_num) + 'intrin_imu.npz', fx=depth_intrin.fx, fy=depth_intrin.fy,
             ppx=depth_intrin.ppx,
             ppy=depth_intrin.ppy,
             coeffs=depth_intrin.coeffs, height=depth_intrin.height, width=depth_intrin.width,
             model=depth_intrin.model,
             imu=angels, allow_pickle=True)  # allow_pickle=True 会调用pickle模块，如果不用，可以考虑将angles数据类型改变
    '''
    保存案例
    0depth_np.npy
    0rgb_img.png
    0intrin_imu.npz
    '''


# Load files data
def rs_data_load(file_num, load_pah="./realsense_data/"):
    '''

    :param file_num: 前面的序号值（一般从0开始）
    :param load_pah: 数据所在的文件夹
    :return: rgb图像，深度数据（numpy格式），相机内参，imu数据[yaw,pitch,roll]
    '''
    # image 文件夹
    # depth 文件夹
    # 内参 文件夹
    img_dir = load_pah + 'image/'
    dep_dir = load_pah + 'depth/'
    int_dir = load_pah + 'intrin/'
    rgb = cv2.imread(img_dir + str(file_num) + 'rgb_img.png')
    depth_image_np = np.load(dep_dir + str(file_num) + 'depth_np.npy')
    get_cameraInfo = np.load(int_dir + str(file_num) + 'intrin_imu.npz', allow_pickle=True)
    # 转化相机内参
    camera_intrin = rs.intrinsics()
    camera_intrin.coeffs = get_cameraInfo['coeffs']
    camera_intrin.fx = get_cameraInfo['fx']
    camera_intrin.fy = get_cameraInfo['fy']
    camera_intrin.ppx = get_cameraInfo['ppx']
    camera_intrin.ppy = get_cameraInfo['ppy']
    camera_intrin.height = get_cameraInfo['height']
    camera_intrin.width = get_cameraInfo['width']
    # imu数据
    imu = get_cameraInfo['imu']
    return rgb, depth_image_np, camera_intrin, imu, img_dir, dep_dir, int_dir


# 输入两个像素点坐标，返回这两个点所确定直线上所有的像素点坐标(返回值代表这条直线的线宽为3个像素)，但实质上并不是直线，跟像素点上画圆一个道理
# 输入坐标只能是按图片位置上的从左到右，坐标点1（x1， y1）一定要在坐标点2（x2， y2)的左侧，否则无法计算
def calculate_slope(c1, c2):
    '''

    :param c1: 起始点
    :param c2: 终止点
    :return: 需要划线的点集列表
    '''
    x1, y1 = c1
    x2, y2 = c2
    if (x2 - x1) == 0:
        print('斜率不存在')
    a = (y2 - y1) / (x2 - x1)
    b = y1 - x1 * ((y2 - y1) / (x2 - x1))
    line_piexl = []
    for i in range(int(x2)):
        if i <= int(x1):
            continue
        elif i > int(x1) & i <= int(x2):
            y = int(a * i + b)
            line_piexl.append([i, y])  # 原直线
            line_piexl.append([i, y - 1])  # 直线向上平移一个像素
            line_piexl.append([i, y + 1])  # 直线向下平移一个像素
    line_piexl = np.array(line_piexl)
    return line_piexl


# 返回两点之间的像素深度
def slope_depth(c1, c2, depth_image_np):
    line_get = calculate_slope(c1, c2)
    depth_line = []
    for line_index, line_pix in enumerate(line_get):
        depth_line.append(depth_image_np[line_pix[1]][line_pix[0]])
    return depth_line


# 通过数学方式获取单个轴旋转矩阵 欧拉角 默认按照pitch角度即y轴旋转
def rotate_mat_pitch(rand_axis=np.array([1, 0, 0]), pitch=0.0):
    '''

    :param rand_axis: 需要旋转的轴
    :param pitch: 旋转角度
    :return:
    '''
    # 获取旋转矩阵
    rot_matrix = linalg.expm(np.cross(np.eye(3), rand_axis / linalg.norm(rand_axis) * pitch))
    return rot_matrix


# 通过open3d获取完整的旋转矩阵
def rotate_mat_all(angles):
    # R = o3d.geometry.get_rotation_matrix_from_yzx([0, roll, pitch]).T
    R = o3d.geometry.get_rotation_matrix_from_yzx(angles).T
    return R


# 将三维地图转化为2维地图
def three2tow(ground_get, map_info, amplify=1, radius=1):
    obstacle_grid_coord_x, obstacle_grid_coord_y = (ground_get[:, 0] * amplify).astype(int), (
            ground_get[:, 1] * amplify).astype(int)
    # x_max, y_max, x_min, y_min = map_info
    x_min, x_max, y_min, y_max = map_info
    # x_max, y_max, x_min, y_min = np.max(obstacle_grid_coord_x), np.max(obstacle_grid_coord_y), \
    #     np.min(obstacle_grid_coord_x), np.min(obstacle_grid_coord_y)
    y_min = 0
    obstacle_grid_coord_x -= x_min
    obstacle_grid_coord_y -= y_min

    w = x_max - x_min + 1
    h = y_max - y_min + 1
    map = np.zeros((h, w), dtype=np.uint8)
    map[obstacle_grid_coord_y, obstacle_grid_coord_x] = 255
    # kernel = np.ones((3, 3), np.uint8)
    # map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel, iterations=3)

    # kernel_dilate = np.ones((grid_size * 2, grid_size * 2), np.uint8)  # 作图用膨胀核
    # map = cv2.dilate(map, kernel_dilate, 2)

    kernel = np.ones((int(radius * 1), int(radius * 1)), np.uint8)  # FIXME 在真实环境中需要进一步限制障碍物
    # map is real map and maps is navigation maps
    maps = cv2.dilate(map, kernel)
    return maps


# 创建局部地图
# def build_local_map(rgbd_imu_pipeline, show=False):
def build_local_map(depth, angles, show=False, intrin=None, filtration=True, height_min=-1.9, height_max=1.5, amplify=1,
                    radius=1):
    '''

    :param depth: 深度数据 可能是numpy类型，也可是pyrealsense2.pyrealsense2.depth_frame类型
    :param pitch: 旋转的角度
    :param show: (bool)是否可视化
    :param intrin: (array)外参矩阵
    :param filtration: (bool)是否过滤
    :param amplify: (float)尺度变化参数
    :return: (np.array)平面化后的地图数据
    :intrin: 如果传入是的numpy类型，需要将内参传入，如果是pyrealsense2.pyrealsense2.depth_frame类型，可以不用（其自带了内参）
    '''
    # print("type depth:", type(depth))
    # height_min = -1.5  # 低于此的可能是地面（与机器人高度相关），过滤掉
    # height_max = 1.9  # 高于此的可能是天花板，过滤掉
    # print("type of depth:", type(depth).__name__)
    if isinstance(depth, rs.depth_frame):  # 通过realsnese官方函数转化点云数据
        warnings.warn(
            "如果有多个角度旋转,建议通过传入numpy uint16类型的深度数据(如直接读取保存的np文件)进行转化.当前使用方法效率较低,未进行滤波操作")
        pc = rs.pointcloud()  # 获取点云句柄
        rotate_pitch = rotate_mat_pitch(np.array([1, 0, 0]), angles[2])  # 获取旋转pitch角度的旋转矩阵
        rotate_roll = rotate_mat_pitch(np.array([0, 0, 1]), angles[1])  # 获取旋转roll角度的旋转矩阵
        points = pc.calculate(depth)  # 深度数据转化为点云数据
        # 获取顶点坐标
        v, t = points.get_vertices(), points.get_texture_coordinates()
        vtx = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        npy_vtx = np.matmul(vtx, rotate_roll)
        map_vtx = np.matmul(npy_vtx, rotate_pitch)  # 坐标转换（乘旋转矩阵）
    elif isinstance(depth, np.ndarray):  # 通过o3d的方式转化数据
        # depth_test = depth.copy()
        R = rotate_mat_all(angles)
        depth = np.asarray(depth, order="C")
        depth = o3d.geometry.Image(depth)

        # 如果intrin的类型为rs.pyrealsense2.intrinsics 则需要转化为o3d的内参
        if type(intrin) is rs.pyrealsense2.intrinsics:
            cam = o3d.camera.PinholeCameraIntrinsic(intrin.width, intrin.height, intrin.fx, intrin.fy, intrin.ppx,
                                                    intrin.ppy)  # 相机内参
        else:
            cam = intrin

        pcd_get = o3d.geometry.PointCloud.create_from_depth_image(depth, cam)

        pcd_get = pcd_get.voxel_down_sample(voxel_size=0.1)  # 点云压缩
        # 统计滤波
        nb_neighbors = 30  # 邻域球内的最少点数，低于该值的点为噪声点
        std_ratio = 2.0  # 标准差倍数
        pcd_get, ind = pcd_get.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                          std_ratio=std_ratio)  # 执行统计滤波，返回滤波后的点云ror_pcd和对应的索引ind
        pcd_get.rotate(R, center=[0, 0, 0])  # o3d中坐标轴， 红色x 绿色y 蓝色z 右手坐标系 此步转换完成后，为realsense的坐标系
        map_vtx = np.asarray(pcd_get.points)
        # vis
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd_get, coordinate_frame])
    # 过滤高度
    # npy_judge = np.where(
    #     np.logical_and(-map_vtx[:, 1] > height_min, -map_vtx[:, 1] < height_max))  # 注意坐标转化和方向问题，第二维反而是z
    map_change = map_vtx.copy()
    path_2d, map_info_change = None, None
    if filtration:
        # print("map info:",amplify,radius)
        # 对地图数据进行初步处理，并获得对应的地图信息
        map_vtx_judge = np.where((abs(map_vtx[:, 0]) < 6) & (map_vtx[:, 2] < 6))
        map_vtx = map_vtx[map_vtx_judge]
        map_x, map_y = (map_vtx[:, 0] * amplify).astype(int), (map_vtx[:, 2] * amplify).astype(int)
        map_info_ori = [np.min(map_x), np.max(map_x), np.min(map_y), np.max(map_y)]

        # 过滤高度和范围 获得障碍物层
        npy_judge = np.where((map_vtx[:, 1] < height_max) & (map_vtx[:, 1] > height_min))  # 注意坐标转化和方向问题，第二维反而是z

        map_change = map_vtx[npy_judge]  # 过滤层
        obstacle_2d = three2tow(map_change[:, [0, 2]], map_info_ori, amplify=amplify, radius=radius)

        # 获取地面的点云
        map_ground = np.where((map_vtx[:, 1] > height_max + 0.1))
        ground_point = map_vtx[map_ground]
        ground_2d = three2tow(ground_point[:, [0, 2]], map_info_ori, amplify=amplify, radius=radius)
        # 地面路面提取
        ground_2d = cv2.bitwise_not(ground_2d)  # 地面取反
        tri_x0 = int(0 - map_info_ori[0])
        tri_y1 = int(map_info_ori[2]) + 50
        # print("map_info_ori is:", map_info_ori)
        triangle_cnt = np.array(
            [(tri_x0, 0), (tri_x0 + tri_y1, tri_y1), (tri_x0 - tri_y1, tri_y1)])  # 绘制出视野三角形（后续再被障碍物修改）
        cv2.drawContours(ground_2d, [triangle_cnt], 0, (0, 0, 0), -1)
        path_2d = ground_2d + obstacle_2d  # 障碍物与地面数据融合

        #  y min一定要置零，因为观察到的障碍物会与机器人本体有一定距离
        map_info_change = map_info_ori
        map_info_change[2] = 0

    # else:
    #     map_vtx = map_vtx
    # map_vtx = map_vtx[:, [0, 2]]  # 取x和y坐标
    # 添加坐标轴 测试用
    # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([axis_pcd])
    print("build_local_map show is:", show)
    if show:
        # 3D坐标
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd_get, coordinate_frame])
        # 使用matplotlib进行绘制3D点云图像
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(map_change[:, 0], map_change[:, 1], map_change[:, 2], s=5)

        # 3D 显示
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(map_change)
        # coordinate_frame 和 pcd合并
        # pcb_all = pcd + coordinate_frame
        # 保存pcd点云
        # pcd_point_save = o3d.io.write_point_cloud("map_change_v1.ply", pcb_all)

        # print("pcd num is:",len(map_vtx))
        # 显示特定点
        # [x1, y1] = [271, 145]
        # [x1, y1] = [276, 280]
        # [x1, y1] = [177, 130]
        # [x1, y1] = [187, 292]
        # [y1, x1] = [271, 145]
        # xyz_test = pixel2xyz_get(depth_test, cam, [x1, y1])
        # R_change = rotate_mat_all([0,0,-np.pi/6])
        # xyz_test @= R_change
        # print("Building map is xyz_test:", xyz_test)
        # 将点在三维地图中显示
        # ax.scatter(xyz_test[0], xyz_test[1], xyz_test[2], s=100, c='r', marker='o')
        # 显示压缩降维地图
        fig = plt.figure()
        # 在原点绘制xyz坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax = fig.add_subplot(111)
        # ax.scatter(np.array(map_vtx[:, 0]), np.array(map_vtx[:, 2]), s=5)  # 散点图，可调整s大小决定膨胀效果
        # map_vtx = map_vtx[:, [0, 2]]
        # print("map_vtx.shape:", map_vtx.shape)
        ax.scatter(map_change[:, 0], map_change[:, 2], s=5)
        # plt.ion()  # 交互式显示
        # plt 非阻塞显示
        # plt.show(block=False)

        # 在图中显示点
        # ax.scatter(xyz_test[0], xyz_test[2], s=100, c='r', marker='o')

        plt.show()
        # 显示点云
        # o3d.visualization.draw_geometries([pcd, axis_pcd])
        # o3d.visualization.draw_geometries([pcd, coordinate_frame])
    # else:
    #     map_vtx = map_vtx[:, [0, 2]]

    return path_2d, map_info_change

# 显示特定点（如识别点）在三维地图中的位置
def show_point_in_map(map_change, xyz_test):
    depth_test = depth.copy()
    R = rotate_mat_all(angles)
    depth = o3d.geometry.Image(depth)
    # 如果intrin的类型为rs.pyrealsense2.intrinsics 则需要转化为o3d的内参
    if type(intrin) is rs.pyrealsense2.intrinsics:
        cam = o3d.camera.PinholeCameraIntrinsic(intrin.width, intrin.height, intrin.fx, intrin.fy, intrin.ppx,
                                                intrin.ppy)  # 相机内参
    else:
        cam = intrin

    pcd_get = o3d.geometry.PointCloud.create_from_depth_image(depth, cam)

    pcd_get = pcd_get.voxel_down_sample(voxel_size=0.1)  # 点云压缩
    # 统计滤波
    nb_neighbors = 30  # 邻域球内的最少点数，低于该值的点为噪声点
    std_ratio = 2.0  # 标准差倍数
    pcd_get, ind = pcd_get.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                      std_ratio=std_ratio)  # 执行统计滤波，返回滤波后的点云ror_pcd和对应的索引ind
    pcd_get.rotate(R, center=[0, 0, 0])  # o3d中坐标轴， 红色x 绿色y 蓝色z 右手坐标系 此步转换完成后，为realsense的坐标系
    map_vtx = np.asarray(pcd_get.points)
    # vis
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_get, coordinate_frame])
    # 过滤高度
    # npy_judge = np.where(
    #     np.logical_and(-map_vtx[:, 1] > height_min, -map_vtx[:, 1] < height_max))  # 注意坐标转化和方向问题，第二维反而是z


    map_change = map_vtx.copy()
    # if show:


# 制作点云文本可视化
def make_point_cloud(center):
    # pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(center)
    # colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    # cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


# 场景图3D可视化
def Scene_diagram_instantiation(object_list):
    import open3d.visualization.gui as gui
    # 创建半径为1的、具有更高分辨率的球体
    radius = 0.1
    resolution = 50
    obj_instances = []  # 需要显示的实例
    # 添加坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
    merged_mesh = coordinate_frame

    # 设置窗口大小、窗口标题等信息
    app = gui.Application.instance
    app.initialize()
    # 文本可视化
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    # vis.add_geometry("Points", points)

    for obj_index in range(len(object_list)):
        print("object_list[obj_index]:", object_list[obj_index])
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        # 位移
        mesh_sphere.translate(object_list[obj_index][0])
        # 上色
        if object_list[obj_index][1] == "chair":  # 椅子红色
            mesh_sphere.paint_uniform_color([0.9, 0.1, 0.1])
        if object_list[obj_index][1] == "table":  # 桌子绿色
            mesh_sphere.paint_uniform_color([0.1, 0.9, 0.1])
        if object_list[obj_index][1] == "blackboard":  # 黑板浅紫色
            mesh_sphere.paint_uniform_color([0.9, 0.1, 0.9])
        if object_list[obj_index][1] == "bed":  # 床棕色
            mesh_sphere.paint_uniform_color([0.5, 0.5, 0.1])
        if object_list[obj_index][1] == "TV":  # 电视黄色
            mesh_sphere.paint_uniform_color([0.5, 0.1, 0.5])
        if object_list[obj_index][1] == "cabinet":  # 柜子青色
            mesh_sphere.paint_uniform_color([0.1, 0.5, 0.5])
        if object_list[obj_index][1] == "door":  # 门白色
            mesh_sphere.paint_uniform_color([0.5, 0.5, 0.5])

        # points = make_point_cloud(100, (0, 0, 0), 1.0)
        # cloud = o3d.geometry.PointCloud()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(object_list[obj_index][0])
        vis.add_3d_label(object_list[obj_index][0], object_list[obj_index][1])
        obj_instances.append(mesh_sphere)
        # 添加文本
        # 创建点云对象

    # 所有实例融合
    for obj_i in obj_instances:
        merged_mesh += obj_i
    # 显示3D效果
    vis.add_geometry("merged_mesh", merged_mesh)
    app.add_window(vis)
    app.run()


# 深度数据可视化
def depth_visualize(depth_img):
    plt.imshow(depth_img, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # pipeline, align, profile = RealScecse_init("213522250540", w=1280, h=800, f=30, camera="D455", mode=2) # 213522250540 043422251232
    # 如果是建图使用, 可以自定义提高分辨率
    # pipeline, align = RealScecse_init_with_imu_single("213522250540", select_custom=True)
    pipeline, align = RealScecse_init_with_imu_single("108222250322", select_custom=True, imu_num=250)

    # rgbd_pipeline, align_color, imu_pipeline, align_acc = RealScecse_init_with_imu("213622074696",imu_get=False)
    # pipeline = align_color
    # align = imu_pipeline
    # pipeline,align = RealScecse_init_with_GPT("108222250322")

    # rgbd_pipeline, align_color, imu_profile, align_acc = RealScecse_init_with_imu("213522250540", imu_get=True)
    # tips：前几帧放掉，让realsense进行自动调整曝光等参数，使得后续数据稳定些
    for i in range(5):
        rgb, aligned_depth_frame, depth_intrin, angles = get_aligned_rgbd_imu_single(pipeline, align, fill_holl=False,
                                                                                     show=False)  # 建议使用函数，可选择是否空洞补齐
        # pipeline, align = RealScecse_init_with_GPT("108222250322")
        # pipeline, align = RealScecse_init_with_imu_single("213522250540", select_custom=True)

        # rgb, aligned_depth_frame, depth_intrin, angles = get_aligned_rgbd_imu_single(pipeline, align)
        # rgbd_pipeline, align_color, imu_pipeline, align_acc = RealScecse_init_with_imu("213622074696", imu_get=False)
        # pipeline = align_color
        # align = imu_pipeline
    # 返回必要参数数据
    rgb, aligned_depth_frame, depth_intrin, angles = get_aligned_rgbd_imu_single(pipeline, align, fill_holl=False,
                                                                                 show=False)  # 建议使用函数，可选择是否空洞补齐
    angles[2] *= -1

    # 地图数据测试 #矮小车
    # maps_path_2d, maps_info = build_local_map(aligned_depth_frame, angles, show=True,
    #                                           intrin=depth_intrin, filtration=True,
    #                                           height_min=-0.3, height_max=0.3, amplify=100,
    #                                           radius=35)
    while 1:
        time_star = time.time()
        # 返回完整参数 相机内参、深度参数、彩色图、深度图、齐帧中的depth帧 3通道深度图
        # intr, depth_intrin, rgb, depth, aligned_depth_frame, depth_image_3d, depth_colormap_aligned,angles = get_aligned_images(
        #     pipeline, align, simple_or=False)  # 获取对齐的图像与相机内参
        # 返回必要参数数据
        rgb, aligned_depth_frame, depth_intrin, angles = get_aligned_rgbd_imu_single(pipeline, align, fill_holl=False,
                                                                                     show=False)  # 建议使用函数，可选择是否空洞补齐

        # 双通道获取imu
        # color_image, depth_frame, depth_intrin, angles = get_aligned_rgbd_imu(pipeline, align, imu_profile, align_acc)
        # 获取空间某一点空间坐标测试
        # center_get = (120, 120)  # 像素坐标
        # cv2.circle(rgb, center_get, 4, (231, 212, 32), 1)
        # xyz_test = rs_get_xyz(aligned_depth_frame, depth_intrin, center_get)
        # theta_1 = time.time()
        # print("angles is:", angles)
        # 新方法 推荐
        # local_map = build_local_map(np.asarray(aligned_depth_frame.get_data(), np.uint16), angles, show=False,
        #                             intrin=depth_intrin, filtration=True)
        #
        local_map = build_local_map(aligned_depth_frame, angles, show=False,
                                    intrin=depth_intrin, filtration=True)
        # 老方法
        # local_map = build_local_map(aligned_depth_frame, angles, show=True, intrin=depth_intrin)
        # print("angles:",angles)
        # theta_2 = time.time()
        # print("build map cost time is:", theta_2 - theta_1)

        cv2.imshow('RGB image', rgb)  # 显示彩色图像

        # Show images
        cv2.namedWindow('RGB image', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            print("rgbd数据已存储")
            rs_data_save(rgb, aligned_depth_frame, depth_intrin, angles, output_path='./realsense_data/')
        elif key == ord('f'):
            rgb_output_path = './image_data'
            if not os.path.exists(rgb_output_path):
                os.makedirs(rgb_output_path)
            file_num = int(len(os.listdir(rgb_output_path)))
            cv2.imwrite("./image_data/" + str(file_num) + 'rgb.png', rgb)
            print("图像已存储：", str(file_num) + 'rgb.png')

        # print("type of local_map:", local_map.shape)
        # plt.figure("figure name screenshot")  # 图像窗口名称
        # plt.scatter(local_map[:, 0], local_map[:, 1])
        # plt.title('text title')  # 图像标题
        # plt.show()

        time_end = time.time()
        # print("检测开销", (time_end - time_star))
        # print("检测速度:", 1 / (time_end - time_star))
