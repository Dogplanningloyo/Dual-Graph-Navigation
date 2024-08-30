# -*- utf-8 -*-
# Author lsy from Server_Robot
# 2023-12-27

import matplotlib
import matplotlib.pyplot as plt

import realsense_deal as rsd

from ultralytics import YOLO
import cv2
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from lightglue import viz2d
import networkx as nx
import numpy as np
import os
import pickle
import yaml
from tqdm import tqdm
import warnings

# from embodied_navigation import Robot_Ai

__version__ = 1.5
# 1.0 不完善的测试版本，可以完成基础框架，建图和拓扑图查找，缺少外部知识库
# 1.1 完善功能，调整代码结构，可实现外部知识库的调用，对位置环境测试
# 1.2 完善节点功能
# 1.3 补充对于未识别等特殊情况的处理
# 1.4 使用tensorrt 替换YOLOV8
# 1.5 去除了不必要的变量（如原图、颜色特征）的存储，降低内存使用


matplotlib.use('TkAgg')

NODE_EXISTED = []
RELATIONSHIP_NAME = "reachable"
SPECIAL_NODE = ["door"]
OBSERVED_NODE = []
RECORD_NODE = []
PATH_NODE = []
SPECIAL_OBSERVED = [None]  # 避免一开始没有对应数据导致的空集报错 对于特殊点有意识的观察
SUCCESS_ARRIVE = False  # 是否成功的标志位
SEARCH_MODE = False  # 是否进行全局搜索模式
LINE_WIDTH = 5  # 直线可达的宽度，越宽越敏感
Detect_name_all = ['AlarmClock', 'apple', 'BasketBall', 'Book', 'Cup', 'GarbageCan', 'bowl', 'Bottle', 'cup', 'box',
                   'Knife', 'laptop', 'Microwave', 'pencil', 'pen', 'ArmChair', 'Chair', 'Footstool', 'Stool', 'Bed',
                   'Desk', 'DiningTable', 'Fridge', 'Safe', 'Cabinet', 'Shelf', 'SideTable', 'Television']

Detect_name_interactive = ['AlarmClock', 'apple', 'BasketBall', 'Book', 'Cup', 'GarbageCan', 'bowl', 'Bottle', 'cup',
                           'box', 'Knife', 'laptop', 'Microwave', 'pencil', 'pen']

Detect_name_anchor_point = ['ArmChair', 'Chair', 'Footstool', 'Stool', 'Bed', 'Desk', 'DiningTable', 'Fridge', 'Safe',
                            'Cabinet', 'Shelf', 'SideTable', 'Television']
# Wall 为缺少对应的物体？
ALL_STEP = 0
OBS_PARA = 0.6  # 障碍物距离站比

file_path = 'a_star.yaml'
with open(file_path, 'r', encoding='utf-8') as read:
    data = yaml.load(read, Loader=yaml.FullLoader)  # 配置文件
version = data['version']
amplify = data['amplify']
radius = data['robot_radius']
grid_size = data['grid_size']  # [m]


# 获取指定文件下的文件数量
def get_file_num(path):
    file_num = len(os.listdir(path))
    return file_num


# 存储对应的目标图像
def save_obj(det, im2):
    results = det(im2)
    # print("results:",results)
    img_tmp = im2.copy()
    for result in results:
        # detection
        boxes = result.boxes  # box with xyxy format, (N, 4)
        cls = result.boxes.cls  # cls, (N, 1)
        # print("cls:", cls)

        if boxes:
            [x1, y1, w1, h1] = boxes.xywh.tolist()[0]
            x_1, y_1, x_2, y_2 = boxes.xyxy.tolist()[0]
            y_1, y_2, x_1, x_2 = int(y_1), int(y_2), int(x_1), int(x_2)
            # print("x_1,y_1,x_2,y_2",x_1, y_1, x_2, y_2)
            # print("x_1:",x_1)
            obj_frame = im2[y_1:y_2, x_1:x_2, :]

            cv2.imshow("obj_frame", obj_frame)
            cv2.rectangle(img_tmp, (int(x1 - w1 / 2), int(y1 - h1 / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)),
                          (251, 215, 225), thickness=3)

            cv2.imshow("img2:", img_tmp)
            name = get_file_num(r'./image')
            key = cv2.waitKey(0)
            if key == ord('s'):
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("目标已保存")
            # elif key == ord('c'):

        # cv2.waitKey()

    # print(tensor_vector)
    # print(len(tensor_vector))


# 返回对应的图像数据
def get_obj_info(det, im2):
    # 获取对应的目标图像
    results = det(im2, conf=0.15)
    # print("results:",results)
    result = results[0]
    print("len(results）：", len(results))
    # for result in results:
    # detection
    boxes = result.boxes  # box with xyxy format, (N, 4)
    cls = result.boxes.cls  # cls, (N, 1)
    cls_ls = cls.tolist()

    # print("cls:", cls)
    img_tmp = False
    if boxes:
        print("len(boxes):", len(boxes))
        for obj_i, boxe in enumerate(boxes):
            img_tmp = im2.copy()
            print("boxe:", boxe)

            [x1, y1, w1, h1] = boxes.xywh.tolist()[obj_i]
            x_1, y_1, x_2, y_2 = boxes.xyxy.tolist()[obj_i]
            y_1, y_2, x_1, x_2 = int(y_1), int(y_2), int(x_1), int(x_2)
            obj_frame = im2[y_1:y_2, x_1:x_2, :]

            # 获取目标的特征信息 如颜色，形状，大小等
            # 获取目标的颜色信息

            # 获取目标的形状信息
            # 获取目标的大小信息
            # 获取目标的位置信息

            cv2.rectangle(img_tmp, (int(x1 - w1 / 2), int(y1 - h1 / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)),
                          (251, 215, 225), thickness=3)
            cv2.imshow("obj_frame", obj_frame)
            cv2.imshow("img_tmp", img_tmp)

            key = cv2.waitKey(0)
            name = 5
            if key == ord("y"):
                print("确定选择该图像")
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("Now 目标已保存")

                # cv2.imshow("img2:", img_tmp)
                # key = cv2.waitKey()
                return obj_frame
    return img_tmp


# 创建一个类，用于对特征提取和比较的封装
class FeatureExtractor(object):
    '''
    用于对特征提取和比较的封装
    extractor:特征提取器
    matcher:特征比较器
    det:目标检测器
    '''

    def __init__(self, extractor, matcher, det):
        # 语义提取器
        self.extractor = extractor
        self.matcher = matcher
        self.graph = nx.Graph()
        self.det = det
        # 数据知识库
        self.obj_names = None  # 物体标签
        self.obj_rates = None  # 物体之间的临接矩阵（表明物体之间相近的可能性）
        self.obj_avg_radius = None  # 物体的平均半径
        # 读取映射字典
        self.name_mapping_dic = None
        # 数据初始化
        self.ekb_init()
        # 记录已经观察到的节点
        self.observed_nodes = []
        # 记录已观察的数量
        self.label_all_num = 0
        # 记录与节点高强度相关的节点
        self.near_node = []

        # self.target_node = self.target_info()

        # 目标相关信息
        class target_info(object):
            def __init__(self):
                self.node_id = None
                self.obj_label = None
                self.obj_color = None
                self.obj_feat = None
                self.image = None
                self.xyz = None
                self.around_fea = np.zeros((1, 80))[0]
                self.box = None

            # 重置目标信息
            def set_update(self, node_id, obj_label, obj_color, obj_feat, image, xyz=None):
                self.node_id = node_id
                self.obj_label = obj_label
                self.obj_color = obj_color
                self.obj_feat = obj_feat
                self.image = image
                self.xyz = xyz

                # 目标节点类型为target_info

        self.target_info = target_info()

    # 获取目标物体与其他物体的相关程度
    def ekb_init(self):
        # self.name_mapping_dic = np.load('./EKB/name_mapping.npy', allow_pickle=True).item()
        # with open("EKB/all_obj_name.pkl", "rb") as fp:
        #     self.obj_names = pickle.load(fp)
        # with open("EKB/objs_avg_radius.pkl", "rb") as fp:
        #     self.obj_avg_radius = pickle.load(fp)
        # with open("EKB/near_rate_array.pkl", "rb") as fp:
        #     self.obj_rates = pickle.load(fp)
        self.obj_rates = np.load("EKB/ekb_rate_get.npy")

    # 图谱清除
    def graph_clear(self):
        self.graph.clear()

    # 返回图谱
    def get_graph(self):
        return self.graph

    # 获取目标可能的大小（先验值） # 需要按照结构修改对应的大小情况
    def pre_obj_size(self, det_id):
        try:
            obj_name = self.name_mapping_dic[det_id]
            # print("obj_name is:", obj_name)
            radius_id = self.obj_names.index(obj_name)  # 获取估计物体可能的大小
            obj_size = int(self.obj_avg_radius[radius_id] * amplify / 3)
        except:
            print("该目标不存在，或外部知识库调用出错，进行预估", det_id)
            obj_size = 15
        return obj_size

    # 获取像素点对应的xyz坐标
    def env_pixel2xyz_get(self_, aligned_depth_frame_, depth_intrin_, R_get_, pixel, env_):
        xyz_test = rsd.pixel2xyz_get(aligned_depth_frame_, depth_intrin_, pixel)
        # 根据环境调整坐标系
        xyz_test = xyz_test @ R_get_
        if env_ == 'Real':
            xyz_test = xyz_test
        elif env_ == 'Ai2Thor':
            xyz_test = xyz_test
        elif env_ == 'Habitat':
            xyz_test = xyz_test  # Habitat中的坐标系单位是米m，要转化为毫米mm
            xyz_test = np.array(xyz_test) * 1000
        return xyz_test

    # 返回得到的目标图像列表
    def get_vis_image(self, img, obj_name, aligned_depth_frame_get, depth_intrin_get, R_get, conf=0.15, show=False,
                      env='Habitat'):
        # 获取对应的目标图像
        im2 = img.copy()
        results_all = self.det(img, conf=conf, classes=obj_name, show=show)
        results_all = results_all[0].boxes  # yolov8原版输出内容
        img_info = []

        for result in results_all:
            # result = result.boxes # yolov8原版输出内容
            cls = int(result.cls)  # cls, (N, 1)
            boxes = result.xyxy.cpu().numpy()[0].astype(int)

            center = result.xywh
            # cls = result['cls']
            # boxes = result['box']  # box with xyxy format, (N, 4)
            # center = result['center']

            # for img_index, box in enumerate(boxes):
            # 获取每个目标的框中心点与坐标
            [x1, y1, w1, h1] = center.tolist()[0]
            x1, y1, w1, h1 = int(x1), int(y1), int(w1), int(h1)
            # [x1, y1] = center
            [x_1, y_1, x_2, y_2] = boxes
            # x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
            # y_1, y_2, x_1, x_2 = int(y_1), int(y_2), int(x_1), int(x_2)
            obj_frame = im2[y_1:y_2, x_1:x_2, :]

            # 通过realsense_deal中的函数获得目标中心的空间坐标，并转化坐标系
            cv2.waitKey(1)
            # xyz_test = rsd.rs_get_xyz(aligned_depth_frame_get, depth_intrin_get, [x1, y1])
            xyz_test = self.env_pixel2xyz_get(aligned_depth_frame_get, depth_intrin_get, R_get, [x1, y1], env)
            # xyz_test = rsd.pixel2xyz_get(aligned_depth_frame_get, depth_intrin_get, [x1, y1])
            # # 根据环境调整坐标系
            # xyz_test = xyz_test @ R_get
            # if env == 'Real':
            #     xyz_test = xyz_test
            # elif env == 'Ai2Thor':
            #     xyz_test = xyz_test
            # elif env == 'Habitat':
            #     xyz_test = xyz_test  # Habitat中的坐标系单位是米m，要转化为毫米mm
            #     xyz_test = np.array(xyz_test) * 1000

            # print("xyz_test:", xyz_test)
            # print("R:", R)

            # img_info.append([obj_frame, cls_ls[img_index], xyz_test])
            img_info.append([obj_frame, cls, xyz_test, boxes])
            if show:
                cv2.imshow("obj_frame", obj_frame)
                cv2.waitKey(1)
        # 获取对应的周围节点特征
        # 获得当前视野所有的目标位置
        all_xyz = [info_node_[2] for info_node_ in img_info]
        for node_i_, node_xyz_ in enumerate(all_xyz):
            # 构建这个节点的外部点特征
            around_feature = np.zeros((1, 80))
            around_feature = around_feature[0]
            # 获取当前目标的xyz
            obj_xyz = node_xyz_
            # 计算所有节点与目标的距离
            all_dis = [np.sqrt(sum(np.power((xyz - obj_xyz), 2))) for xyz in all_xyz]
            # print("all_dis:", all_dis)
            # around_feature_all = []
            # 获取所有距离小于1.5m的节点
            for obj_index_, dis_ in enumerate(all_dis):
                label_id = img_info[obj_index_][1]
                # int(img_info[obj_index_][1].tolist()[0])
                # print("dis_ is:",dis_,label_id)
                if dis_ == 0:  # 目标本身，给一个强调
                    dis_weight = 1500
                    # around_feature[label_id] += 1500
                    # print("arountd got 0:", around_feature)
                elif dis_ < 40 * radius:  # 对于其他非目标，如果满足的
                    # label_id = self.graph.nodes[obj_id_all[obj_index_]]['label']
                    dis_weight = dis_
                    # around_feature[label_id] += dis_
                else:
                    # label_id = 0
                    dis_weight = 0
                around_feature[label_id] += dis_weight
            # print("around_feature is:",around_feature)
            img_info[node_i_].append(around_feature)
        # print("img_info:", img_info)
        return img_info

    # 输入图像，得到图像颜色特征，以及对应的特征点
    def get_object_featrue(self, image):

        # feats = self.load_image_cv2tensor(image)
        feats = self.load_image_cv2feat(image)

        # 提取特征点
        # feats = self.extractor.extract(img_tensor)

        return feats

    # （弃用） 输入图像，得到图像颜色特征，以及对应的特征点
    def get_object_featrue_back(self, image, vis=False):
        warnings.warn("该函数已弃用，新的替换函数不再输出图像颜色分布情况", DeprecationWarning)
        assert False, "该函数已弃用，新的替换函数不再输出图像颜色分布情况"

        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 计算图像1的颜色直方图
        # 只统计H和S两个通道
        hist = cv2.calcHist([image1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # 可视化hist
        if vis:
            plt.plot(hist)
            plt.show()

        # 将统计结果转化为张量向量
        tensor_vector = torch.tensor(hist)

        # feats = self.load_image_cv2tensor(image)
        feats = self.load_image_cv2feat(image)

        # 提取特征点
        # feats = self.extractor.extract(img_tensor)

        return tensor_vector, feats

    # 传入对应节点id，查找是否有对应的节点，返回对应的节点
    def find_node(self, label, color_tensor, feats, image_box, show=False):
        # 记录已识别的目标，避免重复提取识别
        match_node = {}
        # 比较类别、图像颜色、特征点
        for node_i, node in enumerate(self.graph.nodes):
            # 比较是否为同一个节点
            if node == "__obj":  # 不识别直接传入的目标本身 要识别实际观察到的节点
                continue
            node_judge = self.compare_node(self.graph.nodes[node]['label'], self.graph.nodes[node]['color'],
                                           self.graph.nodes[node]['features'], self.graph.nodes[node]['img'],
                                           label, color_tensor, feats, image_box, show=True)
            # print("node_judge:", node_judge)
            if node_judge is not False:
                match_node[node] = node_judge

        # 判断如果存在对应的节点
        if match_node:
            # 找到最大的匹配数量
            max_node = max(match_node, key=match_node.get)
            if show:
                self.match_light_glue_from_networkx(feats, self.graph.nodes[max_node]['features'], show=True,
                                                    image0=image_box,
                                                    image1=self.graph.nodes[max_node]['img'])
                cv2.imshow("self.graph.nodes[max_node]", self.graph.nodes[max_node]['img'])
                cv2.waitKey(0)
            # 返回对应的节点
            return max_node

        else:
            print("在已知地图中未找到目标及其路径")
            return False

    # 创建节点
    def create_node_from_image(self, node_get, label, color_tensor, feats, image_box, xyz, around_fea, box):
        # 记录已识别的目标，避免重复提取识别
        match_node_create = {}
        # 比较类别、图像颜色、特征点，记录每个节点的匹配度
        for node_i, node in enumerate(self.graph.nodes):
            # 比较是否为同一个节点
            # print("node:", self.graph.nodes[node])
            # print("self.graph.nodes[node] is",self.graph.nodes[node]['node_id'],node_get)
            node_judge = self.compare_node(self.graph.nodes[node]['label'], self.graph.nodes[node]['color'],
                                           self.graph.nodes[node]['features'], self.graph.nodes[node]['img'],
                                           self.graph.nodes[node]['around_fea'],
                                           label, color_tensor, feats, image_box, around_fea, show=False)
            # print("node_judge_get:", node_judge)
            # 如果可能是是同一个节点，记录对应的节点id
            if node_judge is not False:
                match_node_create[node] = node_judge

        # TODO 过于冗余，需要修改覆盖
        if match_node_create:
            # 选择匹配度最高的节点
            max_node_create = max(match_node_create, key=match_node_create.get)
            nodes_keys = match_node_create.keys()

            # print("节点已经存在，不需要创建")
            # node_judge = self.compare_node(self.graph.nodes[max_node_create]['label'], self.graph.nodes[max_node_create]['color'],
            #                                self.graph.nodes[max_node_create]['features'], self.graph.nodes[max_node_create]['img'],
            #                                label, color_tensor, feats, image_box,show=True)
            # print("max_node_create:", max_node_create)
            # if max_node_create == "__obj":
            if "__obj" in nodes_keys:
                print("该点为最终目标点")  # 刷新其信息并创建对应的节点
                self.graph.add_node(node_get, node_id=node_get, label=label, color=color_tensor, features=feats,
                                    img=image_box, xyz=xyz, around_fea=around_fea, box=box)
                return "__obj"
            # 节点覆盖
            self.graph.add_node(max_node_create, node_id=max_node_create, label=label, color=color_tensor,
                                features=feats, img=image_box, xyz=xyz, around_fea=around_fea, box=box)
            return max_node_create
        else:
            # 创建新节点
            self.graph.add_node(node_get, node_id=node_get, label=label, color=color_tensor, features=feats,
                                img=image_box,
                                xyz=xyz, around_fea=around_fea, box=box)
            return node_get

    # # 比较途中的节点，确定目标节点实际物理id
    # def determine_target_name(self):

    # 获得当前视野中的有效节点（即不重复的节点）
    def get_image_info(self, image_get, aligned_depth_frame, depth_intrin, R_get, conf=0.15, show=True, obj_show=True):
        # yolo_list = [39, 56, 67, 68, 73]  # FIXME需要按照需求修改
        yolo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                     53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                     78, 79
                     ]  # FIXME需要按照需求修改

        show_img = image_get.copy()
        # 获取对应的目标图像
        img_info = self.get_vis_image(image_get, obj_name=yolo_list, aligned_depth_frame_get=aligned_depth_frame,
                                      depth_intrin_get=depth_intrin, R_get=R_get, conf=conf, show=show)
        # 存储节点id列表
        node_id_list = []

        # 将目标图像列表中逐个提取其特征，并保存到networkx中
        for info_i, info in enumerate(img_info):
            img = info[0]
            cls = info[1]
            xyz = info[2]
            box = info[3]
            around = info[4]
            # print("xyz:",xyz)
            # 获取图像特征
            feats = self.get_object_featrue(img)

            # 创建一个node的id索引，保证其不重复
            # node_id = f"{cls}_{info_i}_{xyz[0]}_{xyz[1]}_{xyz[2]}_{color_tensor}"
            # node_id = f"{cls}_{info_i}_{xyz[0]}_{xyz[1]}_{xyz[2]}_{feats}"
            # node_id_name =
            node_id = f'{cls}_{self.label_all_num}'
            self.label_all_num += 1
            # 显示得到的图像
            # cv2.imshow("img", img)
            # img = None
            color_tensor = None
            # 保存特征到networkx中
            node_get = self.create_node_from_image(node_id, cls, color_tensor, feats, img, xyz, around, box)
            if node_get == "__obj":  # 找到最终目标点，直接返回
                print("该点为最终目标点")
                return [node_get]
            if node_get in self.near_node:  # 如果找到相邻的目标点，直接返回
                return [node_get]
            if node_get in node_id_list:  # 如果有重复的点，直接跳过
                continue
            # cv2.waitKey(1)
            node_id_list.append(node_get)
        if obj_show:
            print("node_id_list is:", node_id_list)
            for num_i_, node_identify_ in enumerate(node_id_list):
                node_show = self.graph.nodes[node_identify_]
                node_box_show = node_show['box']
                node_id_show = node_show['node_id']
                cv2.rectangle(show_img, (node_box_show[0], node_box_show[1]), (node_box_show[2], node_box_show[3]),
                              (123, 22, 33), 3)
                cv2.putText(show_img, str(node_id_show), (node_box_show[0], node_box_show[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (23, 151, 155), thickness=3)
            cv2.imshow(f"show_img", show_img)
            cv2.waitKey(0)
        return node_id_list

    # 创建地图关系
    def create_relationships(self, nodes_get, aligned_depth_frame_ori, depth_intrin_ori, angles):
        map_obj_judge, map_info_judge = rsd.build_local_map(aligned_depth_frame_ori,
                                                            angles, show=False,
                                                            intrin=depth_intrin_ori, filtration=True, height_min=-0.3,
                                                            height_max=1.7, amplify=amplify, radius=radius)

        # map_obj_judge, map_info_judge = bot.maps_path_2d, bot.maps_info
        # for node_id_get_first in tqdm(range(len(nodes_get)), desc="Create relationships now"):
        for first_index, first_node in tqdm(enumerate(nodes_get, start=0),
                                            desc="Create relationships now"):  # 从第二个元素开始，起始索引为 1
            # node_1 = self.graph.nodes[first_node]
            for second_index, second_node in tqdm(enumerate(nodes_get[first_index + 1:]),
                                                  desc="Create relationships now"):  # 从第二个元素开始，起始索引为 1
                obj_xyz_first = self.graph.nodes[first_node]['xyz']
                obj_xyz_second = self.graph.nodes[second_node]['xyz']

                # np不能直接使用int类型变量
                obj_xyz_first_2d = [int(obj_xyz_first[0] - map_info_judge[0]),
                                    int(obj_xyz_first[2] - map_info_judge[2] - 20)]
                obj_xyz_second_2d = [int(obj_xyz_second[0] - map_info_judge[0]),
                                     int(obj_xyz_second[2] - map_info_judge[2] - 20)]
                # TODO 需要考虑障碍物的宽度，以及障碍物的形状 考虑是否融合 bot
                r_start = self.pre_obj_size(self.graph.nodes[first_node]['label'])
                r_end = self.pre_obj_size(self.graph.nodes[second_node]['label'])
                pixel_count_get = self.line_scan(map_obj_judge, obj_xyz_first_2d,
                                                 obj_xyz_second_2d, r_start=r_start,
                                                 r_end=r_end)  # pixel_count_get 一定程度上反映了两个物体之间的障碍物数量和复杂程度
                # 创建关系 # 250 的选择可以过滤大部分关联关系，使得房间独立，不过过于独立了
                # print("pixel_count_get:",pixel_count_get)
                if pixel_count_get < 280:  # 490:  # TODO 半动态更新 还缺少对物体变化后的判断 通过读入内存的方式进行提速
                    euclidean_dis = int(np.linalg.norm(np.array(obj_xyz_first_2d) - np.array(obj_xyz_second_2d)))
                    # euclidean_dis = int(euclidean_dis)
                    composite_path_cost = int(
                        euclidean_dis * (1 - OBS_PARA) + pixel_count_get * OBS_PARA)  # neo4j 无法存储np.float64 的数据类型
                    # 创建graph节点关系
                    self.graph.add_edge(first_node, second_node, obstacle_path=pixel_count_get,
                                        euclidean_dis=euclidean_dis, composite_path_cost=composite_path_cost)

    # 特征对比，并得到匹配的特征点数量
    def match_light_glue_from_networkx(self, feats0, feats1, show=False, image0=None, image1=None):
        # match the features
        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        # 输出匹配的点数
        matches_num = len(matches01["matches"])
        # 绘制比较的特征图（使用opencv）
        if show:
            print("成功匹配到的数据量为:", matches_num)
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            axes = viz2d.plot_images([image0, image1])
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

            kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
            viz2d.plot_images([image0, image1])
            viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
            plt.show()
        return matches_num

    # 将传入的图像转化为tensor 最终转化为提取的特征
    def load_image_cv2feat(self, img, cuda=True):
        image = img.transpose((2, 0, 1))  # HxWxC to CxHxW
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float).cuda()
        if cuda:
            image_tensor = image_tensor.cuda()  # move to CUDA
        feats_extra = self.extractor.extract(image_tensor)

        return feats_extra

    # 统计图像颜色
    def get_object_color(self, image, vis=False):
        return 0
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 计算图像1的颜色直方图
        # 只统计H和S两个通道
        hist = cv2.calcHist([image1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # 可视化hist
        if vis:
            plt.plot(hist)
            plt.show()
        # 将统计结果转化为张量向量
        tensor_vector = torch.tensor(hist)
        return tensor_vector

    # FIXME 缺少深度数据 通过输入图像，获得目标节点数据
    def get_obj_from_image(self, image, obj_label_get=None):
        warnings.warn("该函数已弃用，新的替换函数将以图像、深度值为输入。更换为obj_info_from_img()函数",
                      DeprecationWarning)
        assert rgb, "此函数已被禁用，由同类中 obj_info_from_img 函数替换"
        if obj_label_get:
            obj_label = int(obj_label_get)
        else:
            # 获取对应的目标图像
            results = self.det(image, conf=0.25)
            if results == []:
                obj_label = obj_label_get
            else:
                obj_label = results[0]['cls']
        obj_color = self.get_object_color(image, vis=False)
        obj_feat = self.load_image_cv2feat(image)

        # 创建目标节点
        node_id = f"__obj"
        # node_get, node_id = node_get, label = label, color = color_tensor, features = feats, img = image_box, xyz = xyz
        # image = None
        obj_around_fea = np.zeros((1, 80))
        # fixme 后续需要修改，缺少必要参数 around_fea
        self.graph.add_node(node_id, node_id=node_id, label=obj_label, color=obj_color, features=obj_feat, img=image,
                            xyz=[0, 0, 0], around_fea=obj_around_fea[0])
        # 输出该节点详细数据
        return node_id, obj_label, obj_color, obj_feat, image

    # 显示所有节点
    def show_all_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    # 保存graph数据
    def save_graph(self, path=r'./graph_data/graph_data.pkl', show=False):
        # 如果路径不存在，创造对应的文件夹
        if not os.path.exists(r'./graph_data'):
            os.makedirs(r'./graph_data')
        # 保存graph数据
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
        # 选择是否显示读取的数据
        if show:
            # 打印图谱中所有的点数量
            print("图谱中所有的点数量：", len(self.graph.nodes))
            # 将图谱可视化
            nx.draw(self.graph, with_labels=True)
            plt.show()

    # 读取graph数据
    def load_graph(self, path=r'.\graph_data\graph_data.pkl', show=False):
        # 读取graph数据
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
        # 选择是否显示读取的数据
        if show:
            # print("图谱节点信息：", self.graph.nodes)
            # 打印出每个节点具体属性
            # for node in self.graph.nodes:
            # print("节点本身 ：", node)
            # print("节点属性：", self.graph.nodes[node])
            # 打印图谱中所有的点数量
            print("图谱中所有的点数量：", len(self.graph.nodes))
            # 将图谱可视化
            nx.draw(self.graph, with_labels=True)
            plt.show()

    # 计算两点之间的障碍物数量（本质上是计算像素数量）
    def line_scan(self, image_ori, start, end, r_start=10, r_end=10):
        image = image_ori.copy()
        mask = np.zeros_like(image)
        cv2.line(mask, start, end, 255, LINE_WIDTH)
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

    # 尝试寻找到终点的可行路径
    def get_shortest_path(self, star_id_ls, end_id):
        # 计算每个节点到最终节点的最短路径长度
        shortest_path = nx.shortest_path_length(self.graph, target=end_id, weight='composite_path_cost')
        # 显示所有最短路径的图像
        # for node in shortest_path:
        #     # 最短路径长度
        #     print("shortest_path[node]:", shortest_path[node])
        #     cv2.imshow("short_img_get", self.graph.nodes[node]['img'])
        #     cv2.waitKey(1)
        print("star_id_ls get :", star_id_ls)
        # 选择路径最短的节点
        min_node = min(star_id_ls, key=lambda node: shortest_path.get(node, float('inf')))

        # 判断是否存在最短路径
        if min_node in shortest_path:
            shortest_path_get = nx.shortest_path(self.graph, min_node, end_id, weight='composite_path_cost')
            for node_get in shortest_path_get:  # TODO 可优化，不需要每次都显示？
                # print(" node_get node_id:", self.graph.nodes[node_get]["node_id"])
                # 比较两个节点的图像
                # self.compare_node(self.graph.nodes[node_get]['label'], self.graph.nodes[node_get]['color'],
                #                   self.graph.nodes[node_get]['features'],self.graph.nodes[node_get]['img'],
                #                   self.graph.nodes[end_id]['label'], self.graph.nodes[end_id]['color'],
                #                   self.graph.nodes[end_id]['features'],self.graph.nodes[end_id]['img'],show=True)
                cv2.imshow("short_img_get", self.graph.nodes[node_get]['img'])
                cv2.waitKey(0)

            print("存在最短路径:", min_node)
            print("最短路径长度为：", len(shortest_path_get))
            print("最短路径cost数值为:", shortest_path[min_node])
            cv2.imshow("short_img", self.graph.nodes[min_node]["img"])
            cv2.waitKey(1)
            return min_node, shortest_path_get
        else:
            print("不存在最短路径")
            return None, None

    # 没找到路径时，通过概率进行路径判断
    def find_possible_nodes(self, vis_nodes_id_get, obj_label, find_rate=0.01):
        if vis_nodes_id_get == [] or obj_label is None:
            return None, None
        max_rate = 0
        # 获取对应的label及其索引
        obj_name = self.name_mapping_dic[obj_label]  # 找到对应的名字
        target_ekb_index = self.obj_names.index(obj_name)  # 判定对应的数据索引

        possible_obj_id = None
        # 查找可能性 # 先不探索门等特殊节点，留给后续时候看，先把其他的目标物体给搜索了
        for node_get_ in vis_nodes_id_get:
            node_label = self.graph.nodes[node_get_]['label']
            vis_name_get_ = self.name_mapping_dic[int(node_label)]  # 找到对应的名字

            name_id = self.obj_names.index(vis_name_get_)  # 找到对应的外部知识库概率

            possible_rate = self.obj_rates[target_ekb_index][name_id]  # 与最终目标相近概率
            # print("获得的概率为:", possible_rate)
            if SEARCH_MODE:  # 启动全局搜索模式，将所有的目标进行搜索
                # possible_rate += find_rate + 0.01
                possible_rate = 0.5
            # TODO 可以通过代码技巧，重新得可视范围内，可能性最大的目标，从而避免了对所有目标的搜索，提高了效率
            if possible_rate >= find_rate \
                    and vis_name_get_ not in SPECIAL_NODE \
                    and node_get_ not in OBSERVED_NODE \
                    and possible_rate > max_rate:
                max_rate = possible_rate
                # possible_turn_get = turn_i_possible_
                possible_obj_id = node_get_

                # print(f"Find EKB obj is："
                #       f"name:{possible_obj_id} "
                #       f"rate: {max_rate}")

        # 选择合适的决策
        if possible_obj_id is None:  # 如果所有可能性都已经被尝试过（大概率进入了死锁），开始进行特殊锚点试探
            print("已经没有可找的了")
            return
        elif possible_obj_id is not None:  # 如果有找到合适的点，且该目标没有被尝试过，就导航并在终点附近进行探索
            self.obj_id = possible_obj_id
            # print(f"Find EKB obj isname:{possible_obj_id} ")
            # cv2.imshow("possible_obj_id", self.graph.nodes[possible_obj_id]['img'])
            # cv2.waitKey(1)
            return possible_obj_id, max_rate
        print("We need know the possible_obj_id:", possible_obj_id)
        if possible_obj_id == None:  # 如果一点把握没有，换个方向，再找找
            if find_rate <= 0.001:  # 浮点计算，可能在0.1上下浮动
                print("目前视角内无可信与可尝试的目标，将进行强制运动去打破死锁")

    # 使用学习到的EKB FIXME 待更新，更换为graph
    def find_possible_nodes_graph(self, vis_nodes_id_get, obj_label, find_rate=0.01):
        if vis_nodes_id_get == [] or obj_label is None:
            return None, None
        max_rate = 0
        # 获取对应的label及其索引
        # obj_name = self.name_mapping_dic[obj_label]  # 找到对应的名字
        # target_ekb_index = self.obj_names.index(obj_name)  # 判定对应的数据索引

        possible_obj_id = None
        # 查找可能性 # 先不探索门等特殊节点，留给后续时候看，先把其他的目标物体给搜索了
        for node_get_ in vis_nodes_id_get:
            node_label = self.graph.nodes[node_get_]['label']
            # vis_name_get_ = self.name_mapping_dic[int(node_label)]  # 找到对应的名字

            # name_id = self.obj_names.index(vis_name_get_)  # 找到对应的外部知识库概率
            # print("obj_label:",obj_label,"node_label:",node_label)
            possible_rate = self.obj_rates[obj_label][node_label]  # 与最终目标相近概率
            # print("获得的概率为:", possible_rate)
            if SEARCH_MODE:  # 启动全局搜索模式，将所有的目标进行搜索
                # possible_rate += find_rate + 0.01
                possible_rate = 0.5
            # TODO 可以通过代码技巧，重新得可视范围内，可能性最大的目标，从而避免了对所有目标的搜索，提高了效率
            if possible_rate >= find_rate \
                    and node_label not in SPECIAL_NODE \
                    and node_get_ not in self.observed_nodes \
                    and possible_rate > max_rate:
                max_rate = possible_rate
                # possible_turn_get = turn_i_possible_
                possible_obj_id = node_get_
                print("obj_label:", obj_label, "node_label:", node_label)

                # print(f"Find EKB obj is："
                #       f"name:{possible_obj_id} "
                #       f"rate: {max_rate}")

        # 选择合适的决策
        if possible_obj_id is None:  # 如果所有可能性都已经被尝试过（大概率进入了死锁），开始进行特殊锚点试探
            print("已经没有可找的了")
            return None, None
        elif possible_obj_id is not None:  # 如果有找到合适的点，且该目标没有被尝试过，就导航并在终点附近进行探索
            self.obj_id = possible_obj_id
            # print(f"Find EKB obj isname:{possible_obj_id} ")
            # cv2.imshow("possible_obj_id", self.graph.nodes[possible_obj_id]['img'])
            # cv2.waitKey(1)
            return possible_obj_id, max_rate
        print("We need know the possible_obj_id:", possible_obj_id)
        if possible_obj_id == None:  # 如果一点把握没有，换个方向，再找找
            if find_rate <= 0.001:  # 浮点计算，可能在0.1上下浮动
                print("目前视角内无可信与可尝试的目标，将进行强制运动去打破死锁")

    def draw_node(self, highlight_nodes=None, highlight_color='red', highlight_edge_color='green'):
        # 将所有的节点按顺序赋予新的属性，便于作图
        for node_index, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['draw_node_index'] = node_index

        pos = nx.spring_layout(self.graph)  # 选择节点布局算法

        # 默认节点为所有节点，如果有特殊需求，可以指定节点
        if highlight_nodes == None:
            # 获取节点和边的属性作为标签
            node_labels = {node: data['draw_node_index'] for node, data in self.graph.nodes(data=True)}
            edge_labels = {(u, v): data['composite_path_cost'] for u, v, data in self.graph.edges(data=True)}
            # 绘制图形
            nx.draw_networkx(self.graph, pos=pos, with_labels=True, node_color='lightblue', node_size=500,
                             labels=node_labels)
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels, font_size=10)
        else:
            # 绘制指定边属性
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=highlight_nodes, node_color=highlight_color)
            # 绘制特定的边
            highlight_edges = [(node, neighbor) for node in highlight_nodes for neighbor in self.graph.neighbors(node)]
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=highlight_edges, edge_color=highlight_edge_color)

        # 显示图形
        plt.axis('off')
        plt.show()

    # 返回对应的图像数据
    def get_obj_frame(self, imget, conf=0.15, obj_name='all'):
        warnings.warn("该函数已弃用，新的替换函数将以图像、深度值为输入。更换为get_vis_image()函数",
                      DeprecationWarning)
        assert False, "此函数已被禁用，由同类中 get_vis_image 函数替换"
        # 获取对应的目标图像
        results_all = self.det(imget, conf=conf, classes=obj_name)
        img_tmp = imget.copy()
        # if results: # 如果目标存在
        for result in results_all:
            # result = results
            cls = result['cls']
            boxes = result['box']  # box with xyxy format, (N, 4)

            [x_1, y_1, x_2, y_2] = boxes
            obj_frame = img_tmp[y_1:y_2, x_1:x_2, :]

            # 通过realsense_deal中的函数获得目标中心的空间坐标，并转化坐标系
            cv2.waitKey()
            # xyz_test = rsd.rs_get_xyz(aligned_depth_frame_get, depth_intrin_get, [x1, y1])
            # xyz_test @= R_get

            # cv2.rectangle(img_tmp, (int(x1 - w1 / 2), int(y1 - h1 / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)),
            #               (251, 215, 225), thickness=3)
            cv2.rectangle(img_tmp, (x_1, y_1), (x_2, y_2), (251, 215, 225), thickness=3)
            cv2.imshow("obj_frame", obj_frame)
            cv2.imshow("img_tmp", img_tmp)

            key = cv2.waitKey(0)
            if not os.path.exists(r'./image'):
                os.makedirs(r'./image')
            name = get_file_num(r'./image') + 1
            if key == ord("s"):
                print("确定选择该图像，存储数量为", name)
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("Now 目标已保存:", path_name)
                print("保存的识别标签为：", cls)

            elif key == ord('y'):
                # 打印对应的识别标签
                print("识别标签为：", cls)
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("Now 目标保存并执行选择该图像", path_name)
                return obj_frame

        return img_tmp

    # 对两张图进行匹配
    def match_light_glue_2_img(self, image0=None, image1=None, show=False):
        # 比较颜色上的差距
        color_img0 = self.get_object_color(image0, vis=False)
        color_img1 = self.get_object_color(image1, vis=False)
        color_dis = torch.dist(color_img0, color_img1, p=2)
        print("颜色差距为：", color_dis)
        # 提取特征
        feats0 = self.load_image_cv2feat(image0)
        feats1 = self.load_image_cv2feat(image1)
        # 匹配特征
        matches_num = self.match_light_glue_from_networkx(feats0, feats1, show=show, image0=image0, image1=image1)
        return matches_num

    # 返回目标节点的xz数值，如果不存在，返回None
    def get_obj_xz(self, obj_node):
        if obj_node == None:
            warnings.warn("目标节点不存在，请核对目标")
            return None
        obj_node_exist = self.graph.has_node(obj_node)
        if obj_node_exist:
            node_xz = [self.graph.nodes[obj_node]['xyz'][0] * (amplify / 1000),
                       self.graph.nodes[obj_node]['xyz'][2] * (amplify / 1000)]  # 原来是mm，现在转化为amplify
            return node_xz
        else:
            warnings.warn("目标节点不存在，请核对目标")
            return None

    # TODO 还未使用将目标附近1.5m范围内的节点记录在self.observed_nodes中
    def save_observed_nodes(self, obj_node, vis_node, radius=1500):
        if obj_node == None:
            warnings.warn("目标节点不存在，无法记录，请核对目标")
            return None
        # 获取当前目标的xyz
        obj_xyz = self.graph.nodes[obj_node]['xyz']
        # 获取可视范围内所有节点的xyz
        all_xyz = [self.graph.nodes[node]['xyz'] for node in vis_node]
        # 计算所有节点与目标的距离
        all_dis = [np.sqrt(sum(np.power((xyz - obj_xyz), 2))) for xyz in all_xyz]
        # print("all_dis:", all_dis)
        # 获取所有距离小于1.5m的节点
        observed_nodes = [vis_node[node] for node, dis in enumerate(all_dis) if dis < radius]
        # 将所有节点存储到observed_nodes中
        self.observed_nodes.append(observed_nodes[0])
        # print("Get self.observed_nodes:", self.observed_nodes)
        # print("The num of self.observed_nodes:", len(self.observed_nodes))

    # 比较两个点是否一致
    def compare_node(self, label_0, color_tensor_0, feats_0, image_box_0, around_fea0, label_1, color_tensor_1, feats_1,
                     image_box_1, around_fea1, show=False):
        # 比较类别
        # if label_0 == label_1:
        if True:
            # 比较周围目标的存在情况
            euc_distance_get = np.linalg.norm(around_fea0 - around_fea1)
            # print(label_0, label_1, "euc_distance_get:", euc_distance_get)
            if euc_distance_get < 2000:
                # print("euc_distance_get is:", euc_distance_get)
                # 比较特征点数量 通过颜色等方式，有效降低特征点匹配次数，提高匹配速度
                try:
                    matches_num = self.match_light_glue_from_networkx(feats_0, feats_1, image0=image_box_0,
                                                                      image1=image_box_1, show=False)
                except RuntimeError as e:
                    print("匹配失败，可能是特征点数量不足", e)
                    print("feats_0", feats_0)
                    print("feats_1", feats_1)
                    matches_num = 0
                except IndexError as e:
                    print("匹配失败，维度不匹配", e)

                # print("matches_num:", matches_num)
                if matches_num > 50:
                    # print("matches_num is:", matches_num)
                    # show = True
                    if show:
                        # 显示成功匹配的效果
                        matches_num = self.match_light_glue_from_networkx(feats_0, feats_1, image0=image_box_0,
                                                                          image1=image_box_1, show=True)
                        print("匹配成功，为同一目标:", matches_num)
                        # if around_choose_dis
                    return matches_num
        return False

    # # 用于单独显式目标
    # def show_obj_node(self, img_get):
    #     results_all = self.det(img_get)

    # 比较周围的目标情况
    # def compare_around(self, ar_1, ar_2):
    #     print("对比周围节点情况")

    # 判断是否成功达到目标
    def check_arrive(self, obj_label, obj_color, obj_feat, image, node_get_obj):
        # 判断是否已经到达目标
        node_judge = self.compare_node(self.graph.nodes[node_get_obj]['label'], self.graph.nodes[node_get_obj]['color'],
                                       self.graph.nodes[node_get_obj]['features'],
                                       self.graph.nodes[node_get_obj]['img'],
                                       obj_label, obj_color, obj_feat, image, show=False)
        if node_judge is not False:
            print("目标已达成")
            # 可视化结果
            temp_img = cv2.resize(self.graph.nodes[node_get_obj]['img'], (640, 480))
            cv2.imshow("image_complete", temp_img)
            cv2.waitKey(1)
            node_judge = self.compare_node(self.graph.nodes[node_get_obj]['label'],
                                           self.graph.nodes[node_get_obj]['color'],
                                           self.graph.nodes[node_get_obj]['features'],
                                           self.graph.nodes[node_get_obj]['img'],
                                           obj_label, obj_color, obj_feat, image, show=True)
            print("成功匹配点数为node_judge:", node_judge)
            return True
        else:
            return False

    # 将周围目标作为特征输入 TODO 可优化，减少循环次数
    def feature_around(self, obj_id_all):
        print("around feature get")
        # node_get, node_id = node_get, label = label, color = color_tensor, features = feats,
        # img = image_box, xyz = xyz
        # 定义所有的目标特征集合
        around_feature_all = []
        # 获取可视范围内所有节点的xyz
        all_xyz = [self.graph.nodes[node]['xyz'] for node in obj_id_all]
        for node_get in obj_id_all:
            # 构建这个节点的外部点特征
            around_feature = np.zeros((1, 80))
            around_feature = around_feature[0]
            # 获取当前目标的xyz
            obj_xyz = self.graph.nodes[node_get]['xyz']
            # 计算所有节点与目标的距离
            all_dis = [np.sqrt(sum(np.power((xyz - obj_xyz), 2))) for xyz in all_xyz]
            # print("all_dis:", all_dis)
            # around_feature_all = []
            # 获取所有距离小于1.5m的节点
            for obj_index_, dis_ in enumerate(all_dis):
                # print("dis is:",dis_)
                if dis_ == 0:  # 目标本身，给一个强调
                    label_id = self.graph.nodes[obj_id_all[obj_index_]]['label']
                    dis_weight = 1500
                    # around_feature[label_id] += 1500
                    # print("arountd got 0:", around_feature)
                elif dis_ < 40 * radius:  # 对于其他非目标，如果满足的
                    label_id = self.graph.nodes[obj_id_all[obj_index_]]['label']
                    dis_weight = dis_
                    # around_feature[label_id] += dis_
                else:
                    label_id = 0
                    dis_weight = 0
                around_feature[label_id] += dis_weight
                # observed_nodes = [obj_id_all[node] for node, dis in enumerate(all_dis) if dis < radius]
            # print("obj_get is:",around_feature)
            around_feature_all.append(around_feature)
        return around_feature_all

    # 比较两个图像的相似度
    def match_2_img(self, img1, img2):
        # 提取特征
        feats0 = self.load_image_cv2feat(img1)
        feats1 = self.load_image_cv2feat(img2)
        # 匹配特征
        matches_num = self.match_light_glue_from_networkx(feats0, feats1, show=True, image0=img1, image1=img2)
        print("matches_num is:", matches_num)
        return matches_num

    # 通过记录的内部知识库，整合并提高内部知识库的效率 rgb_ori, aligned_depth_frame_ori, intrinsic, R_point
    def ikb_promot_ekb(self, graph_dir='graph_data'):
        obj_list = self.det.names
        obj_num = len(obj_list)
        graph_pkl_data = os.listdir(graph_dir)
        ekb_build = np.zeros((obj_num, obj_num))
        ekb_rate_get = np.zeros((obj_num, obj_num))
        for graph_file_name in tqdm(graph_pkl_data, desc="Graph data deal"):
            self.load_graph(path=r'./{}/{}'.format(graph_dir, graph_file_name), show=False)
            # Feature_graph.graph_clear()
            nx_data_get = self.get_graph()

            # 读取图谱边点数据
            for edge in nx_data_get.edges(data=True):
                node1, node2, edge_data = edge
                node1_data = int(nx_data_get.nodes[node1]['label'])
                node2_data = int(nx_data_get.nodes[node2]['label'])
                # 对称矩阵
                ekb_build[node1_data][node2_data] += 1
                ekb_build[node2_data][node1_data] += 1
        # 求取相关概率
        mother_all_get = []
        for rate_num in range(obj_num):
            mother_all_get.append(np.sum(ekb_build[rate_num]))
            for rate_obj in range(obj_num):
                rate_obj_index_rate = ekb_build[rate_num][rate_obj] / np.sum(ekb_build[rate_num])
                ekb_rate_get[rate_num][rate_obj] = rate_obj_index_rate
                ekb_rate_get[rate_obj][rate_num] = rate_obj_index_rate
            # ekb_rate_get[node1_data] = ekb_build[rate_num] / np.sum(ekb_build[rate_num])
        # 保存数据
        np.save("ekb_build.npy", ekb_build)
        np.save("ekb_rate_get.npy", ekb_rate_get)
        print(f"Edge Knowledge Base: {ekb_build}")
        print(f"Edge Knowledge Base Rate: {ekb_rate_get}")
        print("Data deal over")

    # 修改节点相关属性
    def node_info_change(self, node_id_, label_name_, label_value_):
        self.graph.nodes[node_id_][label_name_] = label_value_

    # 修改节点id
    def change_node_id(G, old_id, new_id):
        # Check if old_id exists
        if old_id not in G:
            raise ValueError(f"No node with id {old_id} found in the graph")

        # Check if new_id already exists
        if new_id in G:
            raise ValueError(f"Node with id {new_id} already exists in the graph")

        # Get the adjacency (neighbors) of the old node
        neighbors = list(G.neighbors(old_id))

        # Get the attributes of the old node
        attr_dict = G.nodes[old_id]

        # Add a new node with the new_id and the attributes of the old node
        G.add_node(new_id, **attr_dict)

        # Iterate over the neighbors
        for neighbor in neighbors:
            # Add the edge from the new node to the neighbor
            G.add_edge(new_id, neighbor)

        # Remove the old node
        G.remove_node(old_id)

    # 添加目标节点的相邻节点
    def add_near_node(self, node_id):
        self.near_node.append(node_id)

    # 确定目标图像相关信息
    def obj_info_from_img(self, obj_dir, intrinsic_, R_, angles_ori, conf_set=0.15, num=1, auto_choose=True):
        # 查看目标文件夹是否存在
        save_file_get = "./{}/obj_{}".format(obj_dir, num)
        if not os.path.exists(save_file_get):
            # os.makedirs(save_file_get)
            warnings.warn("目标文件夹不存在，请核对目标")
        obj_agent_state = np.load("{}/agent_state_.npy".format(save_file_get), allow_pickle=True).item()
        img_get = cv2.imread("{}/rgb_save_.jpg".format(save_file_get))
        depth_get = np.load("{}/depth_save_.npy".format(save_file_get))
        obj_global_, det_index_ = obj_agent_state["obj_global"], obj_agent_state["det_index"]
        # obj_global_,det_index_ = obj_agent_state["position"],obj_agent_state["rotation"]
        # depth_img = (depth_get * 255 / 10).astype(np.uint8)
        # depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        # cv2.imshow("img_get", img_get)
        # cv2.imshow("depth_img", depth_img)
        # cv2.waitKey(1)
        # img, obj_name, aligned_depth_frame_get, depth_intrin_get, R_get, conf=0.4, show=False,env='Habitat'

        nodes_info_get = self.get_image_info(img_get, depth_get, intrinsic_, R_, conf=conf_set, show=False,
                                             obj_show=False)
        self.create_relationships(nodes_info_get, depth_get, intrinsic_, angles_ori)
        # self.show_all_graph()
        node_obj_ = "__obj"
        if auto_choose:
            print("auto_choose is:", det_index_)
            node_data = nodes_info_get[det_index_]
            img = self.graph.nodes[node_data]['img']
            # cv2.imshow("img_get_choose", img)
            # choose_key = cv2.waitKey(1)
            # if choose_key == ord('y'):
            attr_dict = self.graph.nodes[node_data]
            self.graph.add_node(node_obj_, **attr_dict)
            print("目标已选定，将相邻节点放入")
            neighbors_is = self.graph.neighbors(node_data)
            for neighbor in neighbors_is:
                self.add_near_node(neighbor)
            img_obj = self.graph.nodes[node_obj_]['img']
            # cv2.imshow("img_get_choose_identify", img_obj)
            # cv2.waitKey(1)
            return obj_global_, det_index_
        else:
            while True:  # 字典在遍历时不能修改，所以需要先遍历，再修改
                # for node_i, node_data in enumerate(self.graph.nodes):
                for node_i, node_data in enumerate(nodes_info_get):
                    print("node_i:", node_i)
                    # 显示得到的图像
                    img = self.graph.nodes[node_data]['img']
                    cv2.imshow("img_get_choose", img)
                    choose_key = cv2.waitKey(0)
                    if choose_key == ord('y'):
                        attr_dict = self.graph.nodes[node_data]
                        self.graph.add_node(node_obj_, **attr_dict)
                        print("目标已选定，将相邻节点放入")
                        neighbors_is = self.graph.neighbors(node_data)
                        for neighbor in neighbors_is:
                            self.add_near_node(neighbor)
                        img_obj = self.graph.nodes[node_obj_]['img']
                        cv2.imshow("img_get_choose_identify", img_obj)
                        cv2.waitKey(1)
                        return obj_global_, det_index_

    # 确定具体目标信息
    def decide_obj_info(self, img_get, depth_get, intrinsic_, R_, conf_set=0.15):
        nodes_info_get = self.get_image_info(img_get, depth_get, intrinsic_, R_, conf=conf_set, show=False,
                                             obj_show=False)

        while True:  # 字典在遍历时不能修改，所以需要先遍历，再修改
            for node_i, node_data in enumerate(nodes_info_get):
                print("node_i_choose is:", node_i, node_data)
                # 显示得到的图像
                img = self.graph.nodes[node_data]['img']
                cv2.imshow("img_get_choose_", img)
                choose_key = cv2.waitKey(0)
                if choose_key == ord('y'):
                    # attr_dict = self.graph.nodes[node_data]
                    # self.graph.add_node(node_obj_, **attr_dict)
                    # position = self.graph.nodes[node_data]['xyz']
                    position_ = self.graph.nodes[node_data]['xyz']
                    print("选定目标位姿和识别顺序:", position_, node_i)
                    cv2.imshow("img_get_choose_identify", img)
                    cv2.waitKey(1)

                    return position_, node_i

    # 采集图像
    def get_image_and_save(self, rgb2cv, depth, num=1, show=True):
        cv2.imwrite("./image/rgb_save_{}.jpg".format(num), rgb2cv)
        np.save("./image/depth_save_{}.npy".format(num), depth)
        if show:
            # 将深度图像归一化到0-255之间
            depth_img = (depth * 255 / np.max(depth)).astype(np.uint8)
            # 使用OpenCV的applyColorMap函数将深度图像转换为彩色图像
            color_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            # 显示彩色图像
            cv2.imshow('Color Depth Image', color_img)
            cv2.imshow("rgb2cv", rgb2cv)


if __name__ == '__main__':

    try_time = 0

    # 加载yolo模型与特征提取器，比较器
    yolo_det = YOLO('weights/yolov8n.pt')

    # 输出模型的识别id与类别
    obj_model_info = yolo_det.names
    print("obj_model_info_all:", obj_model_info)
    # yolo_list = [39, 56, 67, 68, 73]  # FIXME需要按照需求修改

    # 输出对应的类别
    # obj_categories = [obj_model_info[obj_id] for obj_id in yolo_list]
    # print("obj_model_info:", obj_categories)

    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(pretrained='superpoint').eval().cuda()  # load the matcher
    Feature_graph_test = FeatureExtractor(extractor, matcher, yolo_det)

    # 测试比较两个图像的特征点
    img_1_ = cv2.imread(r'image/obj_0/rgb_save.jpg')
    img_2_ = cv2.imread(r'./obs_img_screenshot_09.04.2024.png')

    Feature_graph_test.match_2_img(img_1_, img_2_)

    pipeline, align = rsd.RealScecse_init_with_imu_single("213522250540", select_custom=True, imu_num=200)
    # pipeline, align = rsd.RealScecse_init_with_imu_single("213522250540", select_custom=False)
    rgb, aligned_depth_frame, depth_intrin, angles = rsd.get_aligned_rgbd_imu_single(pipeline, align,
                                                                                     fill_holl=True,
                                                                                     show=False,
                                                                                     np_change=True)  # 建议使用函数，可选择是否空洞补齐

    R = rsd.rotate_mat_all(angles)  # 获得旋转矩阵（理论上可以被优化掉）

    # Feature_graph_test = FeatureExtractor(extractor, matcher, yolo_det)

    # 读取外部知识库
    # Feature_graph_test.load_graph(r'.\graph_data\graph_data_v6.pkl', show=True)

    # Feature_graph_test.get_object_color(rgb, vis=True)

    # 传入目标图像数据
    # obj_frame = get_obj_frame(yolo_det, rgb)  # 获取对应的图像数据
    obj_frame = cv2.imread(r'./image/chair_obj40.jpg')
    # rgb = cv2.imread(r'./image/rgb_test.jpg')
    # obj_frame = Feature_graph_test.get_obj_frame(rgb)
    # obj_frame = img
    # cv2.imshow("obj_frame", obj_frame)
    # cv2.waitKey()
    obj_label = 56
    # obj_node = Feature_graph_test.get_obj_from_image(obj_frame, obj_label)
    node_id, obj_label, obj_color, obj_feat, image = Feature_graph_test.get_obj_from_image(obj_frame, obj_label)
    # 设置目标节点相关数据
    Feature_graph_test.target_info.set_update(node_id, obj_label, obj_color, obj_feat, image)

    obj_get_node = Feature_graph_test.find_node(obj_label, obj_color, obj_feat, image)

    # 从已有的图中进行数据匹配
    # Feature_graph_test.save_feature_to_networkx("test", obj_node['label'], obj_node['color'],obj_node['features'],obj_node['img'])

    while True:
        rgb, aligned_depth_frame_ori, depth_intrin_ori, angles = rsd.get_aligned_rgbd_imu_single(pipeline, align,
                                                                                                 fill_holl=True,
                                                                                                 show=False,
                                                                                                 np_change=True)  # 建议使用函数，可选择是否空洞补齐

        R = rsd.rotate_mat_all(angles)  # 获得旋转矩阵（理论上可以被优化掉）

        cv2.imshow("rgb", rgb)
        key = cv2.waitKey(1)

        vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R, show=False)

        if key == ord('j'):
            # vis_node = Feature_graph_test.get_image_info(rgb)
            vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R)
            print("拓展地图")
            # 获取当前视野中所有的节点
            Feature_graph_test.create_relationships(vis_node, aligned_depth_frame, depth_intrin, angles)
            print("视野中所有的点数量：", len(vis_node))
            Feature_graph_test.save_graph(r'.\graph_data\graph_data_v6.pkl', show=False)
        elif key == ord('k'):
            # vis_node = Feature_graph_test.get_image_info(rgb)
            vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R)

            # 查找目标最短路径
            next_node, all_path = Feature_graph_test.get_shortest_path(vis_node, obj_get_node)
            if next_node:
                print("next_node:", Feature_graph_test.graph.nodes[next_node]['xyz'])
                Feature_graph_test.draw_node(highlight_nodes=all_path, highlight_color='red',
                                             highlight_edge_color='green')
            else:
                print("当前视野的目标中，没有找到最短路径")
        # 通过外部知识库，预估目标情况
        elif key == ord('l'):
            vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R)
            possible_obj_get = Feature_graph_test.find_possible_nodes(vis_node, obj_label, find_rate=0.1)
