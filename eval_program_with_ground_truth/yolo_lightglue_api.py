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
# 1.0 Incomplete test version, can complete the basic framework, build maps and topology map search, lack of external knowledge base
# 1.1 Improve the function, adjust the code structure, can realise the call of external knowledge base, test for location environment.
# 1.2 Improve the node function
# 1.3 Add handling of special cases such as unrecognition
# 1.4 Replace YOLOV8 with tensorrt.
# 1.5 Remove storage of unnecessary variables (e.g. original image, colour features) to reduce memory usage


matplotlib.use('TkAgg')

NODE_EXISTED = []
RELATIONSHIP_NAME = "reachable"
SPECIAL_NODE = ["door"]
OBSERVED_NODE = []
RECORD_NODE = []
PATH_NODE = []
SPECIAL_OBSERVED = [None]  #  
SUCCESS_ARRIVE = False  # 
SEARCH_MODE = False  # 
LINE_WIDTH = 5  # ，
Detect_name_all = ['AlarmClock', 'apple', 'BasketBall', 'Book', 'Cup', 'GarbageCan', 'bowl', 'Bottle', 'cup', 'box',
                   'Knife', 'laptop', 'Microwave', 'pencil', 'pen', 'ArmChair', 'Chair', 'Footstool', 'Stool', 'Bed',
                   'Desk', 'DiningTable', 'Fridge', 'Safe', 'Cabinet', 'Shelf', 'SideTable', 'Television']

Detect_name_interactive = ['AlarmClock', 'apple', 'BasketBall', 'Book', 'Cup', 'GarbageCan', 'bowl', 'Bottle', 'cup',
                           'box', 'Knife', 'laptop', 'Microwave', 'pencil', 'pen']

Detect_name_anchor_point = ['ArmChair', 'Chair', 'Footstool', 'Stool', 'Bed', 'Desk', 'DiningTable', 'Fridge', 'Safe',
                            'Cabinet', 'Shelf', 'SideTable', 'Television']
# Wall ？
ALL_STEP = 0
OBS_PARA = 0.6  # 

file_path = 'a_star.yaml'
with open(file_path, 'r', encoding='utf-8') as read:
    data = yaml.load(read, Loader=yaml.FullLoader)  # 
version = data['version']
amplify = data['amplify']
radius = data['robot_radius']
grid_size = data['grid_size']  # [m]


# 
def get_file_num(path):
    file_num = len(os.listdir(path))
    return file_num


# 
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
                print("")
            # elif key == ord('c'):

        # cv2.waitKey()

    # print(tensor_vector)
    # print(len(tensor_vector))


# 
def get_obj_info(det, im2):
    # 
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

            #  ，，
            # 

            # 
            # 
            # 

            cv2.rectangle(img_tmp, (int(x1 - w1 / 2), int(y1 - h1 / 2)), (int(x1 + w1 / 2), int(y1 + h1 / 2)),
                          (251, 215, 225), thickness=3)
            cv2.imshow("obj_frame", obj_frame)
            cv2.imshow("img_tmp", img_tmp)

            key = cv2.waitKey(0)
            name = 5
            if key == ord("y"):
                print("")
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("Now ")

                # cv2.imshow("img2:", img_tmp)
                # key = cv2.waitKey()
                return obj_frame
    return img_tmp


# ，
class FeatureExtractor(object):
    '''
    
    extractor:
    matcher:
    det:
    '''

    def __init__(self, extractor, matcher, det):
        # 
        self.extractor = extractor
        self.matcher = matcher
        self.graph = nx.Graph()
        self.det = det
        # 
        self.obj_names = None  # 
        self.obj_rates = None  # （）
        self.obj_avg_radius = None  # 
        # 
        self.name_mapping_dic = None
        # 
        self.ekb_init()
        # 
        self.observed_nodes = []
        # 
        self.label_all_num = 0
        # 
        self.near_node = []

        # self.target_node = self.target_info()

        # 
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

            # 
            def set_update(self, node_id, obj_label, obj_color, obj_feat, image, xyz=None):
                self.node_id = node_id
                self.obj_label = obj_label
                self.obj_color = obj_color
                self.obj_feat = obj_feat
                self.image = image
                self.xyz = xyz

                # target_info

        self.target_info = target_info()

    # 
    def ekb_init(self):
        # self.name_mapping_dic = np.load('./EKB/name_mapping.npy', allow_pickle=True).item()
        # with open("EKB/all_obj_name.pkl", "rb") as fp:
        #     self.obj_names = pickle.load(fp)
        # with open("EKB/objs_avg_radius.pkl", "rb") as fp:
        #     self.obj_avg_radius = pickle.load(fp)
        # with open("EKB/near_rate_array.pkl", "rb") as fp:
        #     self.obj_rates = pickle.load(fp)
        self.obj_rates = np.load("EKB/ekb_rate_get.npy")

    # 
    def graph_clear(self):
        self.graph.clear()

    # 
    def get_graph(self):
        return self.graph

    # （） # 
    def pre_obj_size(self, det_id):
        try:
            obj_name = self.name_mapping_dic[det_id]
            # print("obj_name is:", obj_name)
            radius_id = self.obj_names.index(obj_name)  # 
            obj_size = int(self.obj_avg_radius[radius_id] * amplify / 3)
        except:
            print("，，", det_id)
            obj_size = 15
        return obj_size

    # xyz
    def env_pixel2xyz_get(self_, aligned_depth_frame_, depth_intrin_, R_get_, pixel, env_):
        xyz_test = rsd.pixel2xyz_get(aligned_depth_frame_, depth_intrin_, pixel)
        # 
        xyz_test = xyz_test @ R_get_
        if env_ == 'Real':
            xyz_test = xyz_test
        elif env_ == 'Ai2Thor':
            xyz_test = xyz_test
        elif env_ == 'Habitat':
            xyz_test = xyz_test  # Habitatm，mm
            xyz_test = np.array(xyz_test) * 1000
        return xyz_test

    # 
    def get_vis_image(self, img, obj_name, aligned_depth_frame_get, depth_intrin_get, R_get, conf=0.15, show=False,
                      env='Habitat'):
        # 
        im2 = img.copy()
        results_all = self.det(img, conf=conf, classes=obj_name, show=show)
        results_all = results_all[0].boxes  # yolov8
        img_info = []

        for result in results_all:
            # result = result.boxes # yolov8
            cls = int(result.cls)  # cls, (N, 1)
            boxes = result.xyxy.cpu().numpy()[0].astype(int)

            center = result.xywh
            # cls = result['cls']
            # boxes = result['box']  # box with xyxy format, (N, 4)
            # center = result['center']

            # for img_index, box in enumerate(boxes):
            # 
            [x1, y1, w1, h1] = center.tolist()[0]
            x1, y1, w1, h1 = int(x1), int(y1), int(w1), int(h1)
            # [x1, y1] = center
            [x_1, y_1, x_2, y_2] = boxes
            # x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
            # y_1, y_2, x_1, x_2 = int(y_1), int(y_2), int(x_1), int(x_2)
            obj_frame = im2[y_1:y_2, x_1:x_2, :]

            # realsense_deal，
            cv2.waitKey(1)
            # xyz_test = rsd.rs_get_xyz(aligned_depth_frame_get, depth_intrin_get, [x1, y1])
            xyz_test = self.env_pixel2xyz_get(aligned_depth_frame_get, depth_intrin_get, R_get, [x1, y1], env)
            # xyz_test = rsd.pixel2xyz_get(aligned_depth_frame_get, depth_intrin_get, [x1, y1])
            # # 
            # xyz_test = xyz_test @ R_get
            # if env == 'Real':
            #     xyz_test = xyz_test
            # elif env == 'Ai2Thor':
            #     xyz_test = xyz_test
            # elif env == 'Habitat':
            #     xyz_test = xyz_test  # Habitatm，mm
            #     xyz_test = np.array(xyz_test) * 1000

            # print("xyz_test:", xyz_test)
            # print("R:", R)

            # img_info.append([obj_frame, cls_ls[img_index], xyz_test])
            img_info.append([obj_frame, cls, xyz_test, boxes])
            if show:
                cv2.imshow("obj_frame", obj_frame)
                cv2.waitKey(1)
        # 
        # 
        all_xyz = [info_node_[2] for info_node_ in img_info]
        for node_i_, node_xyz_ in enumerate(all_xyz):
            # 
            around_feature = np.zeros((1, 80))
            around_feature = around_feature[0]
            # xyz
            obj_xyz = node_xyz_
            # 
            all_dis = [np.sqrt(sum(np.power((xyz - obj_xyz), 2))) for xyz in all_xyz]
            # print("all_dis:", all_dis)
            # around_feature_all = []
            # 1.5m
            for obj_index_, dis_ in enumerate(all_dis):
                label_id = img_info[obj_index_][1]
                # int(img_info[obj_index_][1].tolist()[0])
                # print("dis_ is:",dis_,label_id)
                if dis_ == 0:  # ，
                    dis_weight = 1500
                    # around_feature[label_id] += 1500
                    # print("arountd got 0:", around_feature)
                elif dis_ < 40 * radius:  # ，
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

    # ，，
    def get_object_featrue(self, image):

        # feats = self.load_image_cv2tensor(image)
        feats = self.load_image_cv2feat(image)

        # 
        # feats = self.extractor.extract(img_tensor)

        return feats

    # （） ，，
    def get_object_featrue_back(self, image, vis=False):
        warnings.warn("，", DeprecationWarning)
        assert False, "，"

        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 1
        # HS
        hist = cv2.calcHist([image1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # hist
        if vis:
            plt.plot(hist)
            plt.show()

        # 
        tensor_vector = torch.tensor(hist)

        # feats = self.load_image_cv2tensor(image)
        feats = self.load_image_cv2feat(image)

        # 
        # feats = self.extractor.extract(img_tensor)

        return tensor_vector, feats

    # id，，
    def find_node(self, label, color_tensor, feats, image_box, show=False):
        # ，
        match_node = {}
        # 、、
        for node_i, node in enumerate(self.graph.nodes):
            # 
            if node == "__obj":  #  
                continue
            node_judge = self.compare_node(self.graph.nodes[node]['label'], self.graph.nodes[node]['color'],
                                           self.graph.nodes[node]['features'], self.graph.nodes[node]['img'],
                                           label, color_tensor, feats, image_box, show=True)
            # print("node_judge:", node_judge)
            if node_judge is not False:
                match_node[node] = node_judge

        # 
        if match_node:
            # 
            max_node = max(match_node, key=match_node.get)
            if show:
                self.match_light_glue_from_networkx(feats, self.graph.nodes[max_node]['features'], show=True,
                                                    image0=image_box,
                                                    image1=self.graph.nodes[max_node]['img'])
                cv2.imshow("self.graph.nodes[max_node]", self.graph.nodes[max_node]['img'])
                cv2.waitKey(0)
            # 
            return max_node

        else:
            print("")
            return False

    # 
    def create_node_from_image(self, node_get, label, color_tensor, feats, image_box, xyz, around_fea, box):
        # ，
        match_node_create = {}
        # 、、，
        for node_i, node in enumerate(self.graph.nodes):
            # 
            # print("node:", self.graph.nodes[node])
            # print("self.graph.nodes[node] is",self.graph.nodes[node]['node_id'],node_get)
            node_judge = self.compare_node(self.graph.nodes[node]['label'], self.graph.nodes[node]['color'],
                                           self.graph.nodes[node]['features'], self.graph.nodes[node]['img'],
                                           self.graph.nodes[node]['around_fea'],
                                           label, color_tensor, feats, image_box, around_fea, show=False)
            # print("node_judge_get:", node_judge)
            # ，id
            if node_judge is not False:
                match_node_create[node] = node_judge

        # TODO ，
        if match_node_create:
            # 
            max_node_create = max(match_node_create, key=match_node_create.get)
            nodes_keys = match_node_create.keys()

            # print("，")
            # node_judge = self.compare_node(self.graph.nodes[max_node_create]['label'], self.graph.nodes[max_node_create]['color'],
            #                                self.graph.nodes[max_node_create]['features'], self.graph.nodes[max_node_create]['img'],
            #                                label, color_tensor, feats, image_box,show=True)
            # print("max_node_create:", max_node_create)
            # if max_node_create == "__obj":
            if "__obj" in nodes_keys:
                print("")  # 
                self.graph.add_node(node_get, node_id=node_get, label=label, color=color_tensor, features=feats,
                                    img=image_box, xyz=xyz, around_fea=around_fea, box=box)
                return "__obj"
            # 
            self.graph.add_node(max_node_create, node_id=max_node_create, label=label, color=color_tensor,
                                features=feats, img=image_box, xyz=xyz, around_fea=around_fea, box=box)
            return max_node_create
        else:
            # 
            self.graph.add_node(node_get, node_id=node_get, label=label, color=color_tensor, features=feats,
                                img=image_box,
                                xyz=xyz, around_fea=around_fea, box=box)
            return node_get

    # # ，id
    # def determine_target_name(self):

    # （）
    def get_image_info(self, image_get, aligned_depth_frame, depth_intrin, R_get, conf=0.15, show=True, obj_show=True):
        # yolo_list = [39, 56, 67, 68, 73]  # FIXME
        yolo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                     28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                     53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                     78, 79
                     ]  # FIXME

        show_img = image_get.copy()
        # 
        img_info = self.get_vis_image(image_get, obj_name=yolo_list, aligned_depth_frame_get=aligned_depth_frame,
                                      depth_intrin_get=depth_intrin, R_get=R_get, conf=conf, show=show)
        # id
        node_id_list = []

        # ，networkx
        for info_i, info in enumerate(img_info):
            img = info[0]
            cls = info[1]
            xyz = info[2]
            box = info[3]
            around = info[4]
            # print("xyz:",xyz)
            # 
            feats = self.get_object_featrue(img)

            # nodeid，
            # node_id = f"{cls}_{info_i}_{xyz[0]}_{xyz[1]}_{xyz[2]}_{color_tensor}"
            # node_id = f"{cls}_{info_i}_{xyz[0]}_{xyz[1]}_{xyz[2]}_{feats}"
            # node_id_name =
            node_id = f'{cls}_{self.label_all_num}'
            self.label_all_num += 1
            # 
            # cv2.imshow("img", img)
            # img = None
            color_tensor = None
            # networkx
            node_get = self.create_node_from_image(node_id, cls, color_tensor, feats, img, xyz, around, box)
            if node_get == "__obj":  # ，
                print("")
                return [node_get]
            if node_get in self.near_node:  # ，
                return [node_get]
            if node_get in node_id_list:  # ，
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

    # 
    def create_relationships(self, nodes_get, aligned_depth_frame_ori, depth_intrin_ori, angles):
        map_obj_judge, map_info_judge = rsd.build_local_map(aligned_depth_frame_ori,
                                                            angles, show=False,
                                                            intrin=depth_intrin_ori, filtration=True, height_min=-0.3,
                                                            height_max=1.7, amplify=amplify, radius=radius)

        # map_obj_judge, map_info_judge = bot.maps_path_2d, bot.maps_info
        # for node_id_get_first in tqdm(range(len(nodes_get)), desc="Create relationships now"):
        for first_index, first_node in tqdm(enumerate(nodes_get, start=0),
                                            desc="Create relationships now"):  # ， 1
            # node_1 = self.graph.nodes[first_node]
            for second_index, second_node in tqdm(enumerate(nodes_get[first_index + 1:]),
                                                  desc="Create relationships now"):  # ， 1
                obj_xyz_first = self.graph.nodes[first_node]['xyz']
                obj_xyz_second = self.graph.nodes[second_node]['xyz']

                # npint
                obj_xyz_first_2d = [int(obj_xyz_first[0] - map_info_judge[0]),
                                    int(obj_xyz_first[2] - map_info_judge[2] - 20)]
                obj_xyz_second_2d = [int(obj_xyz_second[0] - map_info_judge[0]),
                                     int(obj_xyz_second[2] - map_info_judge[2] - 20)]
                # TODO ，  bot
                r_start = self.pre_obj_size(self.graph.nodes[first_node]['label'])
                r_end = self.pre_obj_size(self.graph.nodes[second_node]['label'])
                pixel_count_get = self.line_scan(map_obj_judge, obj_xyz_first_2d,
                                                 obj_xyz_second_2d, r_start=r_start,
                                                 r_end=r_end)  # pixel_count_get 
                #  # 250 ，，
                # print("pixel_count_get:",pixel_count_get)
                if pixel_count_get < 280:  # 490:  # TODO   
                    euclidean_dis = int(np.linalg.norm(np.array(obj_xyz_first_2d) - np.array(obj_xyz_second_2d)))
                    # euclidean_dis = int(euclidean_dis)
                    composite_path_cost = int(
                        euclidean_dis * (1 - OBS_PARA) + pixel_count_get * OBS_PARA)  # neo4j np.float64 
                    # graph
                    self.graph.add_edge(first_node, second_node, obstacle_path=pixel_count_get,
                                        euclidean_dis=euclidean_dis, composite_path_cost=composite_path_cost)

    # ，
    def match_light_glue_from_networkx(self, feats0, feats1, show=False, image0=None, image1=None):
        # match the features
        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        # 
        matches_num = len(matches01["matches"])
        # （opencv）
        if show:
            print(":", matches_num)
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

    # tensor 
    def load_image_cv2feat(self, img, cuda=True):
        image = img.transpose((2, 0, 1))  # HxWxC to CxHxW
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float).cuda()
        if cuda:
            image_tensor = image_tensor.cuda()  # move to CUDA
        feats_extra = self.extractor.extract(image_tensor)

        return feats_extra

    # 
    def get_object_color(self, image, vis=False):
        return 0
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 1
        # HS
        hist = cv2.calcHist([image1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # hist
        if vis:
            plt.plot(hist)
            plt.show()
        # 
        tensor_vector = torch.tensor(hist)
        return tensor_vector

    # FIXME  ，
    def get_obj_from_image(self, image, obj_label_get=None):
        warnings.warn("，、。obj_info_from_img()",
                      DeprecationWarning)
        assert rgb, "， obj_info_from_img "
        if obj_label_get:
            obj_label = int(obj_label_get)
        else:
            # 
            results = self.det(image, conf=0.25)
            if results == []:
                obj_label = obj_label_get
            else:
                obj_label = results[0]['cls']
        obj_color = self.get_object_color(image, vis=False)
        obj_feat = self.load_image_cv2feat(image)

        # 
        node_id = f"__obj"
        # node_get, node_id = node_get, label = label, color = color_tensor, features = feats, img = image_box, xyz = xyz
        # image = None
        obj_around_fea = np.zeros((1, 80))
        # fixme ， around_fea
        self.graph.add_node(node_id, node_id=node_id, label=obj_label, color=obj_color, features=obj_feat, img=image,
                            xyz=[0, 0, 0], around_fea=obj_around_fea[0])
        # 
        return node_id, obj_label, obj_color, obj_feat, image

    # 
    def show_all_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    # graph
    def save_graph(self, path=r'./graph_data/graph_data.pkl', show=False):
        # ，
        if not os.path.exists(r'./graph_data'):
            os.makedirs(r'./graph_data')
        # graph
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
        # 
        if show:
            # 
            print("：", len(self.graph.nodes))
            # 
            nx.draw(self.graph, with_labels=True)
            plt.show()

    # graph
    def load_graph(self, path=r'.\graph_data\graph_data.pkl', show=False):
        # graph
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
        # 
        if show:
            # print("：", self.graph.nodes)
            # 
            # for node in self.graph.nodes:
            # print(" ：", node)
            # print("：", self.graph.nodes[node])
            # 
            print("：", len(self.graph.nodes))
            # 
            nx.draw(self.graph, with_labels=True)
            plt.show()

    # （）
    def line_scan(self, image_ori, start, end, r_start=10, r_end=10):
        image = image_ori.copy()
        mask = np.zeros_like(image)
        cv2.line(mask, start, end, 255, LINE_WIDTH)
        # 
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

    # 
    def get_shortest_path(self, star_id_ls, end_id):
        # 
        shortest_path = nx.shortest_path_length(self.graph, target=end_id, weight='composite_path_cost')
        # 
        # for node in shortest_path:
        #     # 
        #     print("shortest_path[node]:", shortest_path[node])
        #     cv2.imshow("short_img_get", self.graph.nodes[node]['img'])
        #     cv2.waitKey(1)
        print("star_id_ls get :", star_id_ls)
        # 
        min_node = min(star_id_ls, key=lambda node: shortest_path.get(node, float('inf')))

        # 
        if min_node in shortest_path:
            shortest_path_get = nx.shortest_path(self.graph, min_node, end_id, weight='composite_path_cost')
            for node_get in shortest_path_get:  # TODO ，？
                # print(" node_get node_id:", self.graph.nodes[node_get]["node_id"])
                # 
                # self.compare_node(self.graph.nodes[node_get]['label'], self.graph.nodes[node_get]['color'],
                #                   self.graph.nodes[node_get]['features'],self.graph.nodes[node_get]['img'],
                #                   self.graph.nodes[end_id]['label'], self.graph.nodes[end_id]['color'],
                #                   self.graph.nodes[end_id]['features'],self.graph.nodes[end_id]['img'],show=True)
                cv2.imshow("short_img_get", self.graph.nodes[node_get]['img'])
                cv2.waitKey(0)

            print(":", min_node)
            print("：", len(shortest_path_get))
            print("cost:", shortest_path[min_node])
            cv2.imshow("short_img", self.graph.nodes[min_node]["img"])
            cv2.waitKey(1)
            return min_node, shortest_path_get
        else:
            print("")
            return None, None

    # ，
    def find_possible_nodes(self, vis_nodes_id_get, obj_label, find_rate=0.01):
        if vis_nodes_id_get == [] or obj_label is None:
            return None, None
        max_rate = 0
        # label
        obj_name = self.name_mapping_dic[obj_label]  # 
        target_ekb_index = self.obj_names.index(obj_name)  # 

        possible_obj_id = None
        #  # ，，
        for node_get_ in vis_nodes_id_get:
            node_label = self.graph.nodes[node_get_]['label']
            vis_name_get_ = self.name_mapping_dic[int(node_label)]  # 

            name_id = self.obj_names.index(vis_name_get_)  # 

            possible_rate = self.obj_rates[target_ekb_index][name_id]  # 
            # print(":", possible_rate)
            if SEARCH_MODE:  # ，
                # possible_rate += find_rate + 0.01
                possible_rate = 0.5
            # TODO ，，，，
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

        # 
        if possible_obj_id is None:  # （），
            print("")
            return
        elif possible_obj_id is not None:  # ，，
            self.obj_id = possible_obj_id
            # print(f"Find EKB obj isname:{possible_obj_id} ")
            # cv2.imshow("possible_obj_id", self.graph.nodes[possible_obj_id]['img'])
            # cv2.waitKey(1)
            return possible_obj_id, max_rate
        print("We need know the possible_obj_id:", possible_obj_id)
        if possible_obj_id == None:  # ，，
            if find_rate <= 0.001:  # ，0.1
                print("，")

    # EKB FIXME ，graph
    def find_possible_nodes_graph(self, vis_nodes_id_get, obj_label, find_rate=0.01):
        if vis_nodes_id_get == [] or obj_label is None:
            return None, None
        max_rate = 0
        # label
        # obj_name = self.name_mapping_dic[obj_label]  # 
        # target_ekb_index = self.obj_names.index(obj_name)  # 

        possible_obj_id = None
        #  # ，，
        for node_get_ in vis_nodes_id_get:
            node_label = self.graph.nodes[node_get_]['label']
            # vis_name_get_ = self.name_mapping_dic[int(node_label)]  # 

            # name_id = self.obj_names.index(vis_name_get_)  # 
            # print("obj_label:",obj_label,"node_label:",node_label)
            possible_rate = self.obj_rates[obj_label][node_label]  # 
            # print(":", possible_rate)
            if SEARCH_MODE:  # ，
                # possible_rate += find_rate + 0.01
                possible_rate = 0.5
            # TODO ，，，，
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

        # 
        if possible_obj_id is None:  # （），
            print("")
            return None, None
        elif possible_obj_id is not None:  # ，，
            self.obj_id = possible_obj_id
            # print(f"Find EKB obj isname:{possible_obj_id} ")
            # cv2.imshow("possible_obj_id", self.graph.nodes[possible_obj_id]['img'])
            # cv2.waitKey(1)
            return possible_obj_id, max_rate
        print("We need know the possible_obj_id:", possible_obj_id)
        if possible_obj_id == None:  # ，，
            if find_rate <= 0.001:  # ，0.1
                print("，")

    def draw_node(self, highlight_nodes=None, highlight_color='red', highlight_edge_color='green'):
        # ，
        for node_index, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['draw_node_index'] = node_index

        pos = nx.spring_layout(self.graph)  # 

        # ，，
        if highlight_nodes == None:
            # 
            node_labels = {node: data['draw_node_index'] for node, data in self.graph.nodes(data=True)}
            edge_labels = {(u, v): data['composite_path_cost'] for u, v, data in self.graph.edges(data=True)}
            # 
            nx.draw_networkx(self.graph, pos=pos, with_labels=True, node_color='lightblue', node_size=500,
                             labels=node_labels)
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels, font_size=10)
        else:
            # 
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=highlight_nodes, node_color=highlight_color)
            # 
            highlight_edges = [(node, neighbor) for node in highlight_nodes for neighbor in self.graph.neighbors(node)]
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=highlight_edges, edge_color=highlight_edge_color)

        # 
        plt.axis('off')
        plt.show()

    # 
    def get_obj_frame(self, imget, conf=0.15, obj_name='all'):
        warnings.warn("，、。get_vis_image()",
                      DeprecationWarning)
        assert False, "， get_vis_image "
        # 
        results_all = self.det(imget, conf=conf, classes=obj_name)
        img_tmp = imget.copy()
        # if results: # 
        for result in results_all:
            # result = results
            cls = result['cls']
            boxes = result['box']  # box with xyxy format, (N, 4)

            [x_1, y_1, x_2, y_2] = boxes
            obj_frame = img_tmp[y_1:y_2, x_1:x_2, :]

            # realsense_deal，
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
                print("，", name)
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("Now :", path_name)
                print("：", cls)

            elif key == ord('y'):
                # 
                print("：", cls)
                path_name = f"./image/chair_obj{str(name)}.jpg"
                cv2.imwrite(path_name, obj_frame)
                print("Now ", path_name)
                return obj_frame

        return img_tmp

    # 
    def match_light_glue_2_img(self, image0=None, image1=None, show=False):
        # 
        color_img0 = self.get_object_color(image0, vis=False)
        color_img1 = self.get_object_color(image1, vis=False)
        color_dis = torch.dist(color_img0, color_img1, p=2)
        print("：", color_dis)
        # 
        feats0 = self.load_image_cv2feat(image0)
        feats1 = self.load_image_cv2feat(image1)
        # 
        matches_num = self.match_light_glue_from_networkx(feats0, feats1, show=show, image0=image0, image1=image1)
        return matches_num

    # xz，，None
    def get_obj_xz(self, obj_node):
        if obj_node == None:
            warnings.warn("，")
            return None
        obj_node_exist = self.graph.has_node(obj_node)
        if obj_node_exist:
            node_xz = [self.graph.nodes[obj_node]['xyz'][0] * (amplify / 1000),
                       self.graph.nodes[obj_node]['xyz'][2] * (amplify / 1000)]  # mm，amplify
            return node_xz
        else:
            warnings.warn("，")
            return None

    # TODO 1.5mself.observed_nodes
    def save_observed_nodes(self, obj_node, vis_node, radius=1500):
        if obj_node == None:
            warnings.warn("，，")
            return None
        # xyz
        obj_xyz = self.graph.nodes[obj_node]['xyz']
        # xyz
        all_xyz = [self.graph.nodes[node]['xyz'] for node in vis_node]
        # 
        all_dis = [np.sqrt(sum(np.power((xyz - obj_xyz), 2))) for xyz in all_xyz]
        # print("all_dis:", all_dis)
        # 1.5m
        observed_nodes = [vis_node[node] for node, dis in enumerate(all_dis) if dis < radius]
        # observed_nodes
        self.observed_nodes.append(observed_nodes[0])
        # print("Get self.observed_nodes:", self.observed_nodes)
        # print("The num of self.observed_nodes:", len(self.observed_nodes))

    # 
    def compare_node(self, label_0, color_tensor_0, feats_0, image_box_0, around_fea0, label_1, color_tensor_1, feats_1,
                     image_box_1, around_fea1, show=False):
        # 
        # if label_0 == label_1:
        if True:
            # 
            euc_distance_get = np.linalg.norm(around_fea0 - around_fea1)
            # print(label_0, label_1, "euc_distance_get:", euc_distance_get)
            if euc_distance_get < 2000:
                # print("euc_distance_get is:", euc_distance_get)
                #  ，，
                try:
                    matches_num = self.match_light_glue_from_networkx(feats_0, feats_1, image0=image_box_0,
                                                                      image1=image_box_1, show=False)
                except RuntimeError as e:
                    print("，", e)
                    print("feats_0", feats_0)
                    print("feats_1", feats_1)
                    matches_num = 0
                except IndexError as e:
                    print("，", e)

                # print("matches_num:", matches_num)
                if matches_num > 50:
                    # print("matches_num is:", matches_num)
                    # show = True
                    if show:
                        # 
                        matches_num = self.match_light_glue_from_networkx(feats_0, feats_1, image0=image_box_0,
                                                                          image1=image_box_1, show=True)
                        print("，:", matches_num)
                        # if around_choose_dis
                    return matches_num
        return False

    # # 
    # def show_obj_node(self, img_get):
    #     results_all = self.det(img_get)

    # 
    # def compare_around(self, ar_1, ar_2):
    #     print("")

    # 
    def check_arrive(self, obj_label, obj_color, obj_feat, image, node_get_obj):
        # 
        node_judge = self.compare_node(self.graph.nodes[node_get_obj]['label'], self.graph.nodes[node_get_obj]['color'],
                                       self.graph.nodes[node_get_obj]['features'],
                                       self.graph.nodes[node_get_obj]['img'],
                                       obj_label, obj_color, obj_feat, image, show=False)
        if node_judge is not False:
            print("")
            # 
            temp_img = cv2.resize(self.graph.nodes[node_get_obj]['img'], (640, 480))
            cv2.imshow("image_complete", temp_img)
            cv2.waitKey(1)
            node_judge = self.compare_node(self.graph.nodes[node_get_obj]['label'],
                                           self.graph.nodes[node_get_obj]['color'],
                                           self.graph.nodes[node_get_obj]['features'],
                                           self.graph.nodes[node_get_obj]['img'],
                                           obj_label, obj_color, obj_feat, image, show=True)
            print("node_judge:", node_judge)
            return True
        else:
            return False

    #  TODO ，
    def feature_around(self, obj_id_all):
        print("around feature get")
        # node_get, node_id = node_get, label = label, color = color_tensor, features = feats,
        # img = image_box, xyz = xyz
        # 
        around_feature_all = []
        # xyz
        all_xyz = [self.graph.nodes[node]['xyz'] for node in obj_id_all]
        for node_get in obj_id_all:
            # 
            around_feature = np.zeros((1, 80))
            around_feature = around_feature[0]
            # xyz
            obj_xyz = self.graph.nodes[node_get]['xyz']
            # 
            all_dis = [np.sqrt(sum(np.power((xyz - obj_xyz), 2))) for xyz in all_xyz]
            # print("all_dis:", all_dis)
            # around_feature_all = []
            # 1.5m
            for obj_index_, dis_ in enumerate(all_dis):
                # print("dis is:",dis_)
                if dis_ == 0:  # ，
                    label_id = self.graph.nodes[obj_id_all[obj_index_]]['label']
                    dis_weight = 1500
                    # around_feature[label_id] += 1500
                    # print("arountd got 0:", around_feature)
                elif dis_ < 40 * radius:  # ，
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

    # 
    def match_2_img(self, img1, img2):
        # 
        feats0 = self.load_image_cv2feat(img1)
        feats1 = self.load_image_cv2feat(img2)
        # 
        matches_num = self.match_light_glue_from_networkx(feats0, feats1, show=True, image0=img1, image1=img2)
        print("matches_num is:", matches_num)
        return matches_num

    # ， rgb_ori, aligned_depth_frame_ori, intrinsic, R_point
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

            # 
            for edge in nx_data_get.edges(data=True):
                node1, node2, edge_data = edge
                node1_data = int(nx_data_get.nodes[node1]['label'])
                node2_data = int(nx_data_get.nodes[node2]['label'])
                # 
                if node1_data == node2_data:
                    ekb_build[node1_data][node2_data] += 1
                else:
                    ekb_build[node1_data][node2_data] += 1
                    ekb_build[node2_data][node1_data] += 1

        # 
        mother_all_get = []
        for rate_num in range(obj_num):
            mother_all_get.append(np.sum(ekb_build[rate_num]))
            for rate_obj in range(obj_num):
                rate_obj_index_rate = ekb_build[rate_num][rate_obj] / np.sum(ekb_build[rate_num])
                ekb_rate_get[rate_num][rate_obj] = rate_obj_index_rate

        # 
        np.save("ekb_build.npy", ekb_build)
        np.save("ekb_rate_get.npy", ekb_rate_get)
        print(f"Edge Knowledge Base: {ekb_build}")
        print(f"Edge Knowledge Base Rate: {ekb_rate_get}")
        print("Data deal over")

    # 
    def node_info_change(self, node_id_, label_name_, label_value_):
        self.graph.nodes[node_id_][label_name_] = label_value_

    # id
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

    # 
    def add_near_node(self, node_id):
        self.near_node.append(node_id)

    # 
    def obj_info_from_img(self, obj_dir, intrinsic_, R_, angles_ori, conf_set=0.15, num=1, auto_choose=True):
        # 
        save_file_get = "./{}/obj_{}".format(obj_dir, num)
        if not os.path.exists(save_file_get):
            # os.makedirs(save_file_get)
            warnings.warn("，")
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
            print("，")
            neighbors_is = self.graph.neighbors(node_data)
            for neighbor in neighbors_is:
                self.add_near_node(neighbor)
            img_obj = self.graph.nodes[node_obj_]['img']
            # cv2.imshow("img_get_choose_identify", img_obj)
            # cv2.waitKey(1)
            return obj_global_, det_index_
        else:
            while True:  # ，，
                # for node_i, node_data in enumerate(self.graph.nodes):
                for node_i, node_data in enumerate(nodes_info_get):
                    print("node_i:", node_i)
                    # 
                    img = self.graph.nodes[node_data]['img']
                    cv2.imshow("img_get_choose", img)
                    choose_key = cv2.waitKey(0)
                    if choose_key == ord('y'):
                        attr_dict = self.graph.nodes[node_data]
                        self.graph.add_node(node_obj_, **attr_dict)
                        print("，")
                        neighbors_is = self.graph.neighbors(node_data)
                        for neighbor in neighbors_is:
                            self.add_near_node(neighbor)
                        img_obj = self.graph.nodes[node_obj_]['img']
                        cv2.imshow("img_get_choose_identify", img_obj)
                        cv2.waitKey(1)
                        return obj_global_, det_index_

    # 
    def decide_obj_info(self, img_get, depth_get, intrinsic_, R_, conf_set=0.15):
        nodes_info_get = self.get_image_info(img_get, depth_get, intrinsic_, R_, conf=conf_set, show=False,
                                             obj_show=False)

        while True:  # ，，
            for node_i, node_data in enumerate(nodes_info_get):
                print("node_i_choose is:", node_i, node_data)
                # 
                img = self.graph.nodes[node_data]['img']
                cv2.imshow("img_get_choose_", img)
                choose_key = cv2.waitKey(0)
                if choose_key == ord('y'):
                    # attr_dict = self.graph.nodes[node_data]
                    # self.graph.add_node(node_obj_, **attr_dict)
                    # position = self.graph.nodes[node_data]['xyz']
                    position_ = self.graph.nodes[node_data]['xyz']
                    print(":", position_, node_i)
                    cv2.imshow("img_get_choose_identify", img)
                    cv2.waitKey(1)

                    return position_, node_i

    # 
    def get_image_and_save(self, rgb2cv, depth, num=1, show=True):
        cv2.imwrite("./image/rgb_save_{}.jpg".format(num), rgb2cv)
        np.save("./image/depth_save_{}.npy".format(num), depth)
        if show:
            # 0-255
            depth_img = (depth * 255 / np.max(depth)).astype(np.uint8)
            # OpenCVapplyColorMap
            color_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            # 
            cv2.imshow('Color Depth Image', color_img)
            cv2.imshow("rgb2cv", rgb2cv)


if __name__ == '__main__':

    try_time = 0

    # yolo，
    yolo_det = YOLO('weights/yolov8n.pt')

    # id
    obj_model_info = yolo_det.names
    print("obj_model_info_all:", obj_model_info)
    # yolo_list = [39, 56, 67, 68, 73]  # FIXME

    # 
    # obj_categories = [obj_model_info[obj_id] for obj_id in yolo_list]
    # print("obj_model_info:", obj_categories)

    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(pretrained='superpoint').eval().cuda()  # load the matcher
    Feature_graph_test = FeatureExtractor(extractor, matcher, yolo_det)

    # 
    img_1_ = cv2.imread(r'image/obj_0/rgb_save.jpg')
    img_2_ = cv2.imread(r'./obs_img_screenshot_09.04.2024.png')

    Feature_graph_test.match_2_img(img_1_, img_2_)

    pipeline, align = rsd.RealScecse_init_with_imu_single("213522250540", select_custom=True, imu_num=200)
    # pipeline, align = rsd.RealScecse_init_with_imu_single("213522250540", select_custom=False)
    rgb, aligned_depth_frame, depth_intrin, angles = rsd.get_aligned_rgbd_imu_single(pipeline, align,
                                                                                     fill_holl=True,
                                                                                     show=False,
                                                                                     np_change=True)  # ，

    R = rsd.rotate_mat_all(angles)  # （）

    # Feature_graph_test = FeatureExtractor(extractor, matcher, yolo_det)

    # 
    # Feature_graph_test.load_graph(r'.\graph_data\graph_data_v6.pkl', show=True)

    # Feature_graph_test.get_object_color(rgb, vis=True)

    # 
    # obj_frame = get_obj_frame(yolo_det, rgb)  # 
    obj_frame = cv2.imread(r'./image/chair_obj40.jpg')
    # rgb = cv2.imread(r'./image/rgb_test.jpg')
    # obj_frame = Feature_graph_test.get_obj_frame(rgb)
    # obj_frame = img
    # cv2.imshow("obj_frame", obj_frame)
    # cv2.waitKey()
    obj_label = 56
    # obj_node = Feature_graph_test.get_obj_from_image(obj_frame, obj_label)
    node_id, obj_label, obj_color, obj_feat, image = Feature_graph_test.get_obj_from_image(obj_frame, obj_label)
    # 
    Feature_graph_test.target_info.set_update(node_id, obj_label, obj_color, obj_feat, image)

    obj_get_node = Feature_graph_test.find_node(obj_label, obj_color, obj_feat, image)

    # 
    # Feature_graph_test.save_feature_to_networkx("test", obj_node['label'], obj_node['color'],obj_node['features'],obj_node['img'])

    while True:
        rgb, aligned_depth_frame_ori, depth_intrin_ori, angles = rsd.get_aligned_rgbd_imu_single(pipeline, align,
                                                                                                 fill_holl=True,
                                                                                                 show=False,
                                                                                                 np_change=True)  # ，

        R = rsd.rotate_mat_all(angles)  # （）

        cv2.imshow("rgb", rgb)
        key = cv2.waitKey(1)

        vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R, show=False)

        if key == ord('j'):
            # vis_node = Feature_graph_test.get_image_info(rgb)
            vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R)
            print("")
            # 
            Feature_graph_test.create_relationships(vis_node, aligned_depth_frame, depth_intrin, angles)
            print("：", len(vis_node))
            Feature_graph_test.save_graph(r'.\graph_data\graph_data_v6.pkl', show=False)
        elif key == ord('k'):
            # vis_node = Feature_graph_test.get_image_info(rgb)
            vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R)

            # 
            next_node, all_path = Feature_graph_test.get_shortest_path(vis_node, obj_get_node)
            if next_node:
                print("next_node:", Feature_graph_test.graph.nodes[next_node]['xyz'])
                Feature_graph_test.draw_node(highlight_nodes=all_path, highlight_color='red',
                                             highlight_edge_color='green')
            else:
                print("，")
        # ，
        elif key == ord('l'):
            vis_node = Feature_graph_test.get_image_info(rgb, aligned_depth_frame_ori, depth_intrin_ori, R)
            possible_obj_get = Feature_graph_test.find_possible_nodes(vis_node, obj_label, find_rate=0.1)
