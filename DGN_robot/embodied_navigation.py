# -*- coding: utf-8 -*-
# @Time    :2024/1/5 23:26
# @Author  :LSY Dreampoet
# @SoftWare:PyCharm

import time
import cv2
# import pickle
import yaml
from a_star import AStarPlanner  # 用于同步做图
import numpy as np
import realsense_deal as rsd  # 应用顺序会影响对应的调用情况，realsense最好提前引用（不能在o3d后面引用）
import RobotControl as rc # 服务机器人控制
# import uart_car as uc  # 底盘控制
import warnings
from ultralytics import YOLO
import torch
import matplotlib

matplotlib.use('TkAgg')
import yolo_lightglue_api as yl
from lightglue import LightGlue, SuperPoint

__version__ = 1.0

width_set = 600
height_set = 600
fov = 90
file_path = 'a_star.yaml'
with open(file_path, 'r', encoding='utf-8') as read:
    data = yaml.load(read, Loader=yaml.FullLoader)  # 配置文件
version = data['version']
amplify = data['amplify']
radius = data['robot_radius']
grid_size = data['grid_size']  # [m]

# 读取映射字典
name_mapping_dic = np.load('./EKB/name_mapping.npy', allow_pickle=True).item()


# 初始化场景
class Robot_Ai:
    def __init__(self):
        # 常用常量
        self.RELATIONSHIP_NAME = "reachable"
        self.NO_DETECTION = ["Floor",
                             "Wall"]  # 用于标记不识别的目标 Pencil 容易在虚拟环境中丢失？ai2thor的问题？Watch?Bed? "Candle","Pen","Pencil","Bed","Ladle"
        self.SPECIAL_NODE = ["door"]
        self.NODE_EXISTED = []
        self.OBSERVED_NODE = []
        self.RECORD_NODE = []
        self.PATH_NODE = []
        self.ALL_STEP = 0
        self.SPECIAL_OBSERVED = [None]  # 避免一开始没有对应数据导致的空集报错 对于特殊点有意识的观察
        self.SUCCESS_ARRIVE = False  # 是否成功的标志位
        self.OBS_PARA = 0.6  # 障碍物距离站比
        self.LINE_WIDTH = 5  # 直线可达的宽度，越宽对障碍物越敏感
        self.SEARCH_MODE = False  # 是否进行全局搜索模式

        self.shortest_node_get = None
        self.ai2_final_rot = None
        self.ai2_final_xyz = None
        self.shortest_path_value = None
        # self.controller = controller  # 控制器
        # self.event = self.controller.step(action="Pass")  # 获得初始化的动作
        self.theta = 0
        # self.agent = self.event.metadata['agent']
        # self.objects = self.event.metadata['objects']
        # self.p = None
        self.a_star_get = AStarPlanner([0, 0, 0, 0], grid_size, map_get=None)  # TODO 后续优化性能可考虑不重复建图
        # 用于导航节点选择
        self.navigation_path_node = []  # 记录的导航路径节点 开头和结尾为起始点与终点
        self.navigation_path_obj_id = []  # 记录导航节点的物理id
        self.navigation_index = 0
        self.node_id = 0  # neo4j中节点id
        self.obj_id = None  # ai2thor中目标id
        self.finally_target = None  # ai2thor 中的最终目标
        self.finally_type = None  # 最终目标的类型
        self.finally_xyz = [0, 0, 0]  # ai2thor中的最终坐标
        self.obj_xyz = [0, 0, 0]  # ai2thor中目标位置
        self.success_flag = False  # 标志当前是否成功
        # 地图信息 TODO 筛选删掉比较近的锚点和无障碍
        self.maps_path_2d = None
        self.maps_info = None
        self.maps_robot_pos = None

        # 提取外部知识库信息
        # 只是图谱对应地址与数据收集
        self.node_name = "graph_node.csv"
        self.rele_name = "graph_rele.csv"
        self.relative_data_name = "graph_data.csv"

        # self.obj_names = None  # 物体标签
        # self.obj_rates = None  # 物体之间的临接矩阵（表明物体之间相近的可能性）
        # self.obj_avg_radius = None  # 物体的平均半径
        # self.G = None
        # self.gds = None
        # self.map_graph = None
        # self.matcher = None

    #     # 初始化程序
    #     self.ekb_rates()
    #
    # # 获取目标物体与其他物体的相关程度
    # def ekb_rates(self):
    #     with open("EKB/all_obj_name.pkl", "rb") as fp:
    #         self.obj_names = pickle.load(fp)
    #     with open("EKB/objs_avg_radius.pkl", "rb") as fp:
    #         self.obj_avg_radius = pickle.load(fp)
    #     with open("EKB/near_rate_array.pkl", "rb") as fp:
    #         self.obj_rates = pickle.load(fp)

    def guide(self, begin=None, r_begin=0, goal=None, r_goal=0, mod=False, iterations=10, path_show=False):
        # print('r_goal:', r_goal)
        # self.maps_path_2d, self.maps_info = build_map(self.event)  # 更新地图
        if goal is None:
            goal = [-50, 230]
        if begin is None:
            begin = [0, 0]
        if mod:
            cv_star = [begin[0] - self.maps_info[0], begin[0] - self.maps_info[2]]
            cv_goal = [goal[0] - self.maps_info[0], goal[1] - self.maps_info[2]]
            cv2.circle(self.maps_path_2d, cv_star, r_begin, (0, 0, 0), -1)
            cv2.circle(self.maps_path_2d, (int(cv_goal[0]), int(cv_goal[1])), r_goal, (0, 0, 0), -1)
            # cv2.imshow("path_2d_mod", self.maps_path_2d)
            # cv2.waitKey(1)
        self.a_star_get = AStarPlanner(map_info=self.maps_info, resolution=grid_size, map_get=self.maps_path_2d)
        path, judge = self.a_star_get.a_star_plan(begin, goal, show=path_show)
        # 菱形寻找
        for ite_num in range(1, iterations):
            if not judge:  # 如果未能成功找到目标路径
                # ite_num = guid_num + 1
                # change_size = int((ite_num+1)*grid_size)
                node_list = [[goal[0], goal[1] - grid_size * ite_num]]  # 先把顶点找到
                for find_node_col in range(1, 1 + ite_num):
                    node_list.append(
                        [goal[0] - grid_size * find_node_col, goal[1] - grid_size * (ite_num - find_node_col)])
                    node_list.append(
                        [goal[0] + grid_size * find_node_col, goal[1] - grid_size * (ite_num - find_node_col)])
                for guid_node in node_list:
                    path, judge = self.a_star_get.a_star_plan(begin, guid_node, show=False)  # 对于可能的路径区域进行查找
                    if judge:  # 如果找到了到达目标的路径
                        path, judge = self.a_star_get.a_star_plan(begin, guid_node, show=path_show)  # FXIME 导航效果可优化
                        break
            else:  # 如果找到了目标就退出（或者超过最大搜索范围）
                break

        return path, judge

    # 判断路径是否可行
    def path_judge(self, possible_get, aligned_depth_frame_get, angles_get, depth_intrin_get, iterations_get=3,
                   path_show=False):
        print("type of aligned_depth_frame_get：", type(aligned_depth_frame_get))
        self.maps_path_2d, self.maps_info = rsd.build_local_map(aligned_depth_frame_get, angles_get, show=True,
                                                                intrin=depth_intrin_get, filtration=True,
                                                                height_min=-0.3, height_max=1.7, amplify=amplify,
                                                                radius=radius)
        # 统计地图中非障碍物（可行地面的数量） 统计map numpy中非零的数量
        # map_path_get = np.count_nonzero(self.maps_path_2d)
        cnt_array = np.where(self.maps_path_2d, 0, 1)
        path_go = np.sum(cnt_array)
        # print("path get is:",np.sum(cnt_array))
        # print("map_path_get:", map_path_get)

        print("possible:", possible_get)

        path, judge = self.guide(r_begin=radius, goal=possible_get, r_goal=10,
                                 mod=True, iterations=iterations_get, path_show=path_show)
        return judge,path_go

    # 使用改变后的菱形A*进行路径探索
    def navigate_diamond(self, possible, aligned_depth_frame_get, depth_intrin_get, angles_get, path_show=False):
        self.maps_path_2d, self.maps_info = rsd.build_local_map(aligned_depth_frame_get, angles_get, show=False,
                                                                intrin=depth_intrin_get, filtration=True,
                                                                height_min=-0.3, height_max=1.7, amplify=amplify,
                                                                radius=radius)
        # possible = [Feature_graph.graph.nodes[next_node]['xyz'][0] / 10,
        #             Feature_graph.graph.nodes[next_node]['xyz'][2] / 10]
        print("possible:", possible)

        path, judge = self.guide(r_begin=radius, goal=possible, r_goal=10,
                                 mod=True, path_show=path_show)
        print("path is:", path)
        # 找到转弯拐点 与 连续超过100的点
        prev_vector = path[0]
        turning_points = [prev_vector]  # 起点
        # hundred_flag = 0 # 判断有无超过10的点 意味着连续运动超过了1000mm
        for path_i in range(1, len(path)):
            vector_change = path[path_i] - prev_vector
            # hundred_flag += 1 # 累计标志位

            if vector_change[0] != 0 and vector_change[1] != 0:
                # hundred_flag = 0 # 重置标志位
                prev_vector = path[path_i - 1]
                turning_points.append(prev_vector)

        # turning_points.append(path[-1])  # 终点
        print("turning_points Path Line:", turning_points)

        speedset = [0, 0, 0]
        for path_index in range(len(turning_points) - 1):
            [x, y] = turning_points[path_index + 1] - turning_points[path_index]
            x = int(x)
            y = int(y)

            if y:
                speedset = [y * 10, 0, 0]
            elif x:
                speedset = [0, -x * 10, 0]
            else:
                warnings.warn("Movement error运动缺失(不应该出现)")
                continue
            print("speedset is:", speedset)
            robot_control.chassis.setlocationXYZ(speedset)
            cv2.waitKey(0)


# 角度制转弧度制
def to_rad(th):
    return th * np.pi / 180


# 创建暂时性地图
def build_temp_map(map_array):
    map_vtx = map_array
    map_x, map_y = (map_vtx[:, 0] * amplify).astype(int), (
            map_vtx[:, 2] * amplify).astype(int)
    map_info_ori = [np.min(map_x), np.max(map_x), np.min(map_y), np.max(map_y)]

    map_get = three2tow(map_vtx[:, [0, 2]], map_info_ori)
    #  y min一定要置零，因为观察到的障碍物会与机器人本体有一定距离
    map_info_change = map_info_ori
    map_info_change[2] = 0

    return map_get, map_info_change


# 将三维地图转化为2维地图
def three2tow(ground_get, map_info):
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
    kernel = np.ones((3, 3), np.uint8)
    map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel, iterations=1)

    # kernel_dilate = np.ones((grid_size * 2, grid_size * 2), np.uint8)  # 作图用膨胀核
    # map = cv2.dilate(map, kernel_dilate, 2)

    kernel = np.ones((int(radius * 1), int(radius * 1)), np.uint8)
    # map is real map and maps is navigation maps
    maps = cv2.dilate(map, kernel)
    return maps


# 计算两点之间的障碍物数量（本质上是计算像素数量）
def line_scan(image_ori, start, end, r_start=10, r_end=10):
    image = image_ori.copy()
    mask = np.zeros_like(image)
    cv2.line(mask, start, end, 255, bot.LINE_WIDTH)
    # 给起点和终点画圆
    cv2.circle(image, start, r_start, (0, 0, 0), -1)
    cv2.circle(image, end, r_end, (0, 0, 0), -1)
    # cv2.imshow("path_2d_mod", image)
    # print("start and end:",start,end)
    # print("r_start and r_end:",r_start,r_end)
    line_pixels = cv2.bitwise_and(image, mask)
    pixels_count = np.count_nonzero(line_pixels)
    # cv2.imshow("line_pixels", line_pixels)
    # print("get the linepixels is:",pixels_count)
    # cv2.waitKey(2)
    return pixels_count


if __name__ == '__main__':
    # 程序初始化
    print("Star to real")
    # Load a model
    # obj_list = [56, 63, 66, 62, 64, 73] # 准备识别的目标物体
    # 提取器和对比器初始化定义
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(pretrained='superpoint').eval().cuda()  # load the matcher
    obj_list = None
    yolo_det = YOLO('weights/yolov8m.pt')  # pretrained YOLOv8n model

    # 初始化识别器
    Feature_graph = yl.FeatureExtractor(extractor, matcher, yolo_det)

    weight_dict = torch.load("weights/yolov8m.pt")
    weight_name = weight_dict['model'].names
    print("weight_name:", weight_name)
    # rgb, depth_image_np, camera_intrin, imu, img_dir, dep_dir, int_dir = rsd.rs_data_load(file_num=0)
    # 机器人内外部知识库参数初始化
    bot = Robot_Ai()
    # 绘制导航
    rgb, depth_image_np, camera_intrin, imu, img_dir, dep_dir, int_dir = rsd.rs_data_load(file_num=48)
    imu[2] = -imu[2]
    bot.path_judge([172, 314], depth_image_np, imu,
                   camera_intrin, iterations_get=3, path_show=True)
    # 机器人初始化
    robot_control = rc.RobotControl(port='COM3', threadinginside=True, queue_len=10, keyboardcontrol=False)  # 机器人控制初始化
    robot_control.chassis.setspeed([0, 0, 0])
    robot_control.chassis.setconf(2, 600, 200, 12, 4) # 设置对应的速度参数
    robot_control.chassis.setscreenangle(90, True)  # 屏幕向前
    robot_control.chassis.setlocationXYZ([0,500,0])

    # print("angles is:",angles[1]*180/np.pi,angles[2]*180/np.pi)

    # 底盘初始化
    # car_control = uc.Uart_control(port_move = 'COM3', port_camera = 'COM4')  # 机器人控制初始化
    # 机器人内外部知识库参数初始化
    bot = Robot_Ai()
    # 绘制导航
    rgb, depth_image_np, camera_intrin, imu, img_dir, dep_dir, int_dir = rsd.rs_data_load(file_num=30)
    bot.path_judge([150,240], depth_image_np, imu,
                   camera_intrin, iterations_get=3, path_show=True)
    # print("bot.obj_names:", bot.obj_names)
    # print("pre_obj_size:", pre_obj_size(56))

    pipeline, align = rsd.RealScecse_init_with_imu_single("213522250540", select_custom=True)
    # pipeline, align = rsd.RealScecse_init_with_imu_single("108222250322", select_custom=True,imu_num=250)
    # 获取当前realsense数据
    rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, angles_ori = rsd.get_aligned_rgbd_imu_single(pipeline, align)

    # # tips：前几帧放掉，让realsense进行自动调整曝光等参数，使得后续数据稳定些
    for i in range(5):
        rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, angles_ori = rsd.get_aligned_rgbd_imu_single(pipeline,
                                                                                                         align)

    cen_x, cen_y = 100, 100
    target_ls = 56  # 64 #39 64

    # Feature_graph.get_object_color(rgb, vis=True)

    # obj_frame = cv2.imread(r'./image/chair_obj40.jpg')
    # 比较两张图片的匹配情况
    # img_2_matchs = Feature_graph.match_light_glue_2_img(obj_frame, obj_frame_2,show=True)
    # print("img_2_matches is:",img_2_matchs)
    # 读取内部拓扑地图
    internal_map = r'.\graph_data\graph_data_9.pkl'
    # Feature_graph.load_graph(internal_map, show=True)

    # 传入目标图像数据
    # obj_frame = Feature_graph.get_obj_frame(rgb_ori)  # 获取对应的图像数据
    # 获取最终目标效果
    obj_frame = cv2.imread(r'./image/chair_obj47.jpg')
    obj_lable = 39 # 56
    node_id, obj_lable, obj_color, obj_feat, image = Feature_graph.get_obj_from_image(obj_frame, obj_lable)
    obj_get_node = Feature_graph.find_node(obj_lable, obj_color, obj_feat, image, show=True)

    while 1:
        # posible = [0,0] # 初始化对应的目标点
        # result_get = []
        time_star = time.time()
        # 返回完整参数 相机内参、深度参数、彩色图、深度图、齐帧中的depth帧 3通道深度图
        # intr, depth_intrin, rgb, depth, aligned_depth_frame, depth_image_3d, depth_colormap_aligned,angles = get_aligned_images(
        #     pipeline, align, simple_or=False)  # 获取对齐的图像与相机内参
        # 返回必要参数数据
        rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, angles_ori = rsd.get_aligned_rgbd_imu_single(pipeline,
                                                                                                         align,
                                                                                                         fill_holl=True,
                                                                                                         show=False)  # 建议使用函数，可选择是否空洞补齐
        # 通过存储的数据进行测试
        # rgb, depth_image_np, camera_intrin, imu, img_dir, dep_dir, int_dir = rsd.rs_data_load(file_num=0)
        # print("angles is:",angles[1]*180/np.pi,angles[2]*180/np.pi)
        # angles = imu

        # aligned_depth_frame = depth_image_np
        # depth_intrin = camera_intrin
        angles_ori[2] *= -1
        R = rsd.rotate_mat_all(angles_ori)  # 获得旋转矩阵（理论上可以被优化掉）
        center_get = (cen_x, cen_y)  # 像素坐标
        # cv2.circle(rgb, center_get, 4, (231, 212, 32), 1)
        # xyz_test = rsd.rs_get_xyz(aligned_depth_frame, depth_intrin, center_get)
        cv2.imshow("rgb", rgb_ori)
        key = cv2.waitKey(1)
        # print("self.obj_names:",Feature_graph.obj_names)
        if key == ord('j'):
            # 获取当前视野中所有的节点
            vis_node = Feature_graph.get_image_info(rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, R)
            if vis_node[0] == "__obj":  # 发现最终目标
                print("发现最终目标")
            print("拓展地图")
            # 构建拓扑地图
            Feature_graph.create_relationships(vis_node, aligned_depth_frame_ori, depth_intrin_ori, angles_ori)
            print("视野中所有的点数量：", len(vis_node))
            Feature_graph.save_graph(internal_map, show=True)
        elif key == ord('k'):

            vis_node = Feature_graph.get_image_info(rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, R)
            if vis_node[0] == True:  # 发现最终目标
                print("发现最终目标")
            Feature_graph.create_relationships(vis_node, aligned_depth_frame_ori, depth_intrin_ori, angles_ori)
            print("地图已构建：", len(vis_node))
            Feature_graph.save_graph(internal_map, show=False)

            # 查找目标最短路径
            next_node, all_path = Feature_graph.get_shortest_path(vis_node, obj_get_node)
            if next_node:
                print("next_node:", Feature_graph.graph.nodes[next_node]['xyz'])
                Feature_graph.draw_node(highlight_nodes=all_path, highlight_color='red', highlight_edge_color='green')
                # possible_xz = [Feature_graph.graph.nodes[next_node]['xyz'][0]/10, Feature_graph.graph.nodes[next_node]['xyz'][2]/10]
                next_xz = Feature_graph.get_obj_xz(next_node)
                bot.navigate_diamond(next_xz, aligned_depth_frame_ori, depth_intrin_ori, angles_ori, path_show=True)
                # 判断是否到达目标点
                # node_id, obj_lable, obj_color, obj_feat, image, node_get_obj
                check_arrive = Feature_graph.check_arrive(obj_lable, obj_color, obj_feat, image, obj_get_node)


        elif key == ord('l'):
            possible_turn = {}
            for turn_i in range(3):  # 左前右，无后
                robot_control.chassis.setscreenangle(int(turn_i * 90), True)  # 屏幕向左
                time.sleep(3)  # 等视觉稳定
                rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, angles_ori = rsd.get_aligned_rgbd_imu_single(
                    pipeline, align,
                    fill_holl=True,
                    show=False)  # 建议使用函数，可选择是否空洞补齐
                angles_ori[2] *= -1
                R = rsd.rotate_mat_all(angles_ori)  # 获得旋转矩阵（理论上可以被优化掉）

                cv2.imshow("rgb", rgb_ori)
                cv2.waitKey(1)
                vis_node = Feature_graph.get_image_info(rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, R)
                Feature_graph.create_relationships(vis_node, aligned_depth_frame_ori, depth_intrin_ori, angles_ori)
                print("地图已构建：", len(vis_node))
                if vis_node[0] == "__obj":
                    possible_obj_get = vis_node[0]
                    possible_turn[turn_i] = 11000  # 给出最大的可能性，但后续仍然需要判断是否右可行路径
                    print("发现最终目标，直接导航过去 STOP")
                    # break
                else:
                    possible_obj_get, obj_rate = Feature_graph.find_possible_nodes(vis_node, obj_lable,
                                                                                   find_rate=0.1)  # 对于外部探索，主要给出类别
                    possible_turn[turn_i] = obj_rate

                possible_point_temp = Feature_graph.get_obj_xz(possible_obj_get)
                arrive_judge,path_get = bot.path_judge(possible_point_temp, aligned_depth_frame_ori, angles_ori,
                                              depth_intrin_ori, iterations_get=3, path_show=False)
                if arrive_judge:
                    possible_turn[turn_i] = possible_turn[turn_i]*10000 + path_get  # 如果不可达，直接判负 # TODO 当然，如果都不可达，应该是不动，后续可以增加打破死锁的方案
                    continue
                else:
                    possible_turn[turn_i] = -1*10000 + path_get  # 如果不可达，直接降低其概率权重 # TODO 当然，如果都不可达，应该是不动，后续可以增加打破死锁的方案

            robot_control.chassis.setscreenangle(90, True)  # 屏幕向前
            # 选择最优的方向
            turn_get = max(possible_turn, key=possible_turn.get)
            print("possible_turn:", possible_turn)
            print("turn_get:", turn_get)
            # 地盘运动到指定方向
            if turn_get == 0:  # 向左
                speedset = [0, 0, int(-30000 / 4)]
            elif turn_get == 1:  # 向前
                speedset = [0, 0, 0]
            else:  # 向右
                speedset = [0, 0, int(30000 / 4)]
            robot_control.chassis.setlocationXYZ(speedset)
            time.sleep(3)

            Feature_graph.save_graph(r'.\graph_data\graph_data_v8.pkl', show=False)

            rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, angles_ori = rsd.get_aligned_rgbd_imu_single(
                pipeline, align,
                fill_holl=True,
                show=False)  # 建议使用函数，可选择是否空洞补齐
            angles_ori[2] *= -1
            R = rsd.rotate_mat_all(angles_ori)  # 获得旋转矩阵（理论上可以被优化掉）
            vis_node = Feature_graph.get_image_info(rgb_ori, aligned_depth_frame_ori, depth_intrin_ori, R)

            # Feature_graph.create_relationships(vis_node, aligned_depth_frame_ori, depth_intrin_ori, angles_ori)
            possible_obj_get, obj_rate = Feature_graph.find_possible_nodes(vis_node, obj_lable,
                                                                           find_rate=0.1)  # 对于外部探索，主要给出类别
            if vis_node[0] == "__obj":
                print("选择找到的最终目标，图像为")
                cv2.imshow("obj_get", Feature_graph.graph.nodes[vis_node[0]]['image'])
            possible_point_xz = Feature_graph.get_obj_xz(possible_obj_get)
            print("We get the possible_obj_get xz:", possible_obj_get)
            Feature_graph.save_observed_nodes(possible_obj_get, vis_node)  # 将目标附件的节点进行保存
            bot.navigate_diamond(possible_point_xz, aligned_depth_frame_ori, depth_intrin_ori, angles_ori,
                                 path_show=False)  # FIXME 导航终点判定有些问题
            check_arrive = Feature_graph.check_arrive(obj_lable, obj_color, obj_feat, image, possible_obj_get)

        elif key == ord('q'):
            speedset = [0, 0, int(30000 / 4)]
            robot_control.chassis.setlocationXYZ(speedset)
            # cv2.waitKey(1)
        elif key == ord('e'):
            speedset = [0, 0, int(-30000 / 4)]
            robot_control.chassis.setlocationXYZ(speedset)
            # cv2.waitKey(1)


