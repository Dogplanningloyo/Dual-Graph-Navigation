import logging

logging.basicConfig(level=logging.INFO, filename='generate.log', filemode='w',
                    format='%(asctime)s-%(levelname)s-%(message)s')
import copy
import math
import pickle
import sys
import networkx as nx
from ai2thor.platform import CloudRendering
from ai2thor.controller import Controller
from ai2thor.util.metrics import (path_distance, get_shortest_path_to_object_type, get_shortest_path_to_object,
                                  compute_single_spl, get_shortest_path_to_point)
import prior
import numpy as np
import cv2
import time
import open3d as o3d
from py2neo import Graph, Node, Relationship, NodeMatcher
from graphdatascience import GraphDataScience
from Planning.a_star import AStarPlanner  # A-star
from Planning.RRT import RRT_Planner  # RRT

import yaml
import warnings
from tqdm import tqdm
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import multiprocessing

# Guide with radius and all r_begin and r_goal means radius of something
__version__ = 3.6


# 2.0 Basic testing
# 3.0 Special nodes such as doors (room-to-room observations), with special treatment
# 3.1 Explicit categorisation of test data sets
# 3.2 Create multiple databases for storing different test data
# 3.3 Fixing and avoiding emulator transmission bugs
# 3.4 Start multi-threaded debugging mode
# 3.5 Fixing the drawing of navigation paths
# 3.51 Clarify navigation data division
# 3.6 High concurrency operation
# 3.7 Save and check evaluation data


# Initialise the scenario
class Robotcontrol:
    def __init__(self, house_env):
        controller = Controller(scene=house_env,
                                width=width_set,
                                height=height_set,
                                fieldOfView=fov,
                                grid_size=0.01,
                                visibilityDistance=9,
                                gridSize=grid_size * 0.01,
                                renderDepthImage=True,
                                renderInstanceSegmentation=True,
                                # agentMode="default"
                                platform=CloudRendering,
                                )
        # Common Constants
        self.RELATIONSHIP_NAME = "reachable"
        self.NO_DETECTION = ["Floor",
                             "Wall"]
        self.SPECIAL_NODE = ["door"]
        self.NODE_EXISTED = []
        self.OBSERVED_NODE = []
        self.RECORD_NODE = []
        self.PATH_NODE = []
        # First person view
        self.RECORD_NODE_DRAW = []
        self.ALL_STEP = 0
        self.SPECIAL_OBSERVED = [None]
        self.SUCCESS_ARRIVE = False  # Success flag bit
        self.OBS_PARA = 0.6  # Obstacle distance station ratio
        self.LINE_WIDTH = 5  # The width of the line reachable, the wider the more sensitive to obstacles
        self.SEARCH_MODE = False

        self.shortest_node_get = None
        self.ai2_final_rot = None
        self.ai2_final_xyz = None
        self.shortest_path_value = None
        self.controller = controller  # Controller
        self.event = self.controller.step(action="Pass")
        self.theta = 0
        self.agent = self.event.metadata['agent']
        self.objects = self.event.metadata['objects']
        # self.p = None
        self.a_star_get = AStarPlanner([0, 0, 0, 0], grid_size, map_get=None)  # init A-star
        self.RRT_get = RRT_Planner([0, 0, 0, 0], None)  # init RRT

        # Decision-making parameters
        self.navigation_path_node = []
        self.navigation_path_obj_id = []
        self.navigation_index = 0
        self.node_id = 0
        self.obj_id = None
        self.finally_target = None
        self.finally_type = None
        self.finally_xyz = [0, 0, 0]
        self.obj_xyz = [0, 0, 0]
        self.success_flag = False
        self.cross_observed = []  # IKG Used to record obstacles crossed

        self.maps_path_2d = None
        self.maps_info = None
        self.maps_robot_pos = None

        self.node_name = "graph_node.csv"
        self.rele_name = "graph_rele.csv"
        self.relative_data_name = "graph_data.csv"

        self.obj_names = None  # Object label
        self.obj_rates = None  # Correlation between objects
        self.obj_avg_radius = None  # The mean radius of the object
        self.G = None
        self.gds = None
        self.map_graph = None
        self.matcher = None

        # Init EKG
        self.ekg_init()

        self.save_img_node_num = 0

    def export_neo4j_data(self, dir_neo_ex):
        graph_all_data_path = os.path.join(dir_neo_ex, self.relative_data_name)
        cypher_export = f"CALL apoc.export.csv.all('{graph_all_data_path}', " \
                        "{quotes: 'none', useTypes: false, delim: ';' })"
        # results_export = self.map_graph.run(cypher_export).data()

    def import_neo4j_data(self, dir_neo_im):
        graph_node_file = os.path.join(dir_neo_im, self.node_name)
        graph_rele_file = os.path.join(dir_neo_im, self.rele_name)
        graph_all_data_path = os.path.join(dir_neo_im, self.relative_data_name)

        graph_type = {
            "_id": str,
            "_start": str,
            "_end": str,
        }
        df_graph = pd.read_csv(graph_all_data_path, encoding="utf-8", sep=';', dtype=graph_type)
        graph_node = df_graph[df_graph['_labels'].notna()]
        graph_node = graph_node[['_id', 'Type', 'abs_xyz', 'obj_id']]
        graph_node = graph_node.rename(columns={"_id": ":ID"})

        graph_node.to_csv(graph_node_file, sep='|', index=None)

        graph_rele = df_graph[df_graph['_labels'].isnull()]
        graph_rele = graph_rele[['_start', '_end', 'composite_path_cost', 'euclidean_dis', 'obstacle_path']]
        graph_rele = graph_rele.rename(columns={"_start": ":START_ID", "_end": ":END_ID",
                                                "composite_path_cost": "composite_path_cost:FLOAT",
                                                'euclidean_dis': 'euclidean_dis:FLOAT',
                                                'obstacle_path': 'obstacle_path:FLOAT'})

        graph_rele.to_csv(graph_rele_file, sep='|', index=None)

        # Knowledge graph
        cypher_import = r" CALL apoc.import.csv(" \
                        f"[{{fileName: '{graph_node_file}', labels: ['object']}}]," \
                        f"[{{fileName: '{graph_rele_file}', type: 'reachable'}}], " \
                        r"{delimiter: '|', arrayDelimiter: ',', stringIds: false, ignoreDuplicateNodes: true})"
        results_item = self.map_graph.run(cypher_import).data()

    # load pri EKG
    def ekg_init(self):
        with open("EKB/all_obj_name.pkl", "rb") as fp:
            self.obj_names = pickle.load(fp)
        with open("EKB/objs_avg_radius.pkl", "rb") as fp:
            self.obj_avg_radius = pickle.load(fp)
        with open("EKB/near_rate_array_ekg.pkl", "rb") as fp:
            self.obj_rates = pickle.load(fp)

    # Online update
    def ikb_update_ekb(self, graph_dir='graph_data'):
        obj_num = len(self.obj_names)
        ekb_build = np.zeros((obj_num, obj_num))
        update_rate = 0.3
        query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        data = self.map_graph.run(query)
        for record in data:
            node1_type = record[0]['Type']  # Star node
            relationship = record[1]  # Edge
            node2_type = record[2]['Type']  # End node
            if node1_type not in self.obj_names:
                self.obj_names.append(node1_type)
                rows, cols = self.obj_rates.shape
                new_row = np.zeros((1, cols))
                new_column = np.zeros((rows + 1, 1))
                self.obj_rates = np.append(self.obj_rates, new_row, axis=0)
                self.obj_rates = np.append(self.obj_rates, new_column, axis=1)
                ekb_build = np.append(ekb_build, new_row, axis=0)
                ekb_build = np.append(ekb_build, new_column, axis=1)

            elif node2_type not in self.obj_names:
                self.obj_names.append(node2_type)
                rows, cols = self.obj_rates.shape
                new_row = np.zeros((1, cols))
                new_column = np.zeros((rows + 1, 1))
                self.obj_rates = np.append(self.obj_rates, new_row, axis=0)
                self.obj_rates = np.append(self.obj_rates, new_column, axis=1)
                ekb_build = np.append(ekb_build, new_row, axis=0)
                ekb_build = np.append(ekb_build, new_column, axis=1)

            node1_type_index = self.obj_names.index(node1_type)
            node2_type_index = self.obj_names.index(node2_type)

            if node1_type_index == node2_type_index:
                ekb_build[node1_type_index][node2_type_index] += 1
            else:
                ekb_build[node1_type_index][node2_type_index] += 1
                ekb_build[node2_type_index][node1_type_index] += 1

        # Find the relevant probability
        # possible_rate = self.obj_rates[target_ekb_index][name_id]

        mother_all_get = []
        for rate_num in range(obj_num):
            mother_all_get.append(np.sum(ekb_build[rate_num]))
            for rate_obj in range(obj_num):
                rate_obj_index_rate = ekb_build[rate_num][rate_obj] / np.sum(ekb_build[rate_num])
                self.obj_rates[rate_num][rate_obj] = update_rate * rate_obj_index_rate + (1 - update_rate) * \
                                                     self.obj_rates[rate_num][rate_obj]

    # Navigation with begin and goal's radius
    def guide(self, begin=None, r_begin=0, goal=None, r_goal=0, mod=False, path_show=False):
        # print('r_goal:', r_goal)
        self.maps_path_2d, self.maps_info = build_map()
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
        # self.RRT_get = RRT_Planner(self.maps_info,self.maps_path_2d, step_size=grid_size)
        # path, judge = self.RRT_get.plan(begin, goal,show=path_show)

        # Diamond search
        for ite_num in range(1, 10):
            if not judge:
                # ite_num = guid_num + 1
                # change_size = int((ite_num+1)*grid_size)
                node_list = [[goal[0], goal[1] - grid_size * ite_num]]
                for find_node_col in range(1, 1 + ite_num):
                    node_list.append(
                        [goal[0] - grid_size * find_node_col, goal[1] - grid_size * (ite_num - find_node_col)])
                    node_list.append(
                        [goal[0] + grid_size * find_node_col, goal[1] - grid_size * (ite_num - find_node_col)])
                for guid_node in node_list:
                    path, judge = self.a_star_get.a_star_plan(begin, guid_node, show=path_show)
                    # path, judge = self.RRT_get.plan(begin, goal)
                    if judge:
                        break
            else:
                break

        return path, judge

    # Navigation control motion
    def path_movement(self, path):
        # print("Start motion navigation", self.obj_id)
        for path_index in tqdm(range(len(path) - 1), desc="Moving follow the path:"):
            get_third_img()  # Update the third person view camera

            [x, y] = path[path_index + 1] - path[path_index]
            if y:  # Default go ahead no back
                # self.controller.step("MoveAhead",moveMagnitude=0.2)
                move_event = self.controller.step("MoveAhead")
            elif x > 0:
                # self.controller.step("MoveRight",moveMagnitude=0.2)
                move_event = self.controller.step("MoveRight")
            elif x < 0:
                # self.controller.step("MoveLeft",moveMagnitude=0.2)
                move_event = self.controller.step("MoveLeft")
            else:
                warnings.warn("Movement error")
                continue
            self.ALL_STEP += 1
            self.collect_now_node()

            if plot_experiment_picture:
                check_step(move_event.frame)

        # print("Reach navigation target：", self.obj_id)
        return True

    # Save current coordinates
    def collect_now_node(self):
        self.event = self.controller.step(action="Stand")
        self.agent = self.event.metadata['agent']
        [cv_x_get, cv_y_get, end_x, end_y] = self.real_2_cv_xyz(self.agent['position']['x'],
                                                                self.agent['position']['z'])
        bot.PATH_NODE.append(bot.agent['position'])
        bot.RECORD_NODE.append([cv_x_get, cv_y_get, end_x, end_y])

    # Look down at the pixel coordinates of the image and the actual current position
    def real_2_cv_xyz(self, ai_pos_x, ai_pos_z):
        # bot.PATH_NODE.append(bot.agent['position'])
        # cv_x = 30 * float(ai_pos_x) + 65
        cv_x = 30 * float(ai_pos_x) + 40
        # cv_y = -28 * float(ai_pos_z) + 540
        cv_y = -28 * float(ai_pos_z) + 480
        cv_x_get, cv_y_get = int(cv_x), int(cv_y)
        rot_get = bot.agent['rotation']
        end_x = cv_x_get + int(20 * np.sin(rot_get['y'] / 180 * np.pi))
        end_y = cv_y_get - int(20 * np.cos(rot_get['y'] / 180 * np.pi))
        return [cv_x_get, cv_y_get, end_x, end_y]

    # Draw the shortest path graph
    def the_shortest_path_draw(self, shortest_path):  # cvyz
        third_img_get_ = get_third_img()
        short_img_draw = third_img_get_.copy()
        # short_img_draw = short_img_draw.copy()
        short_line = []
        for short_node in shortest_path:
            [cv_x_get, cv_y_get, cv_x_end_, cv_y_end_] = self.real_2_cv_xyz(short_node['x'], short_node['z'])
            cv2.circle(short_img_draw, (cv_x_get, cv_y_get), 5, (23, 12, 233), 2)
            short_line.append([cv_x_get, cv_y_get])
        short_line = np.array(short_line)
        cv2.polylines(short_img_draw, [short_line], isClosed=False, color=(23, 212, 24), thickness=2)
        # cv2.imshow("short_img_draw", short_img_draw)
        cv2.imwrite(process_sub_epoch + "/short_img_draw.jpg", short_img_draw)
        cv2.imwrite(process_sub_epoch + "/get_third_img.jpg", third_img_get_)

    # The shortest path effect in the drawing process
    def draw_path_node(self, path_node_id, roud):
        path_node_img_ = get_third_img()
        path_node_img_ = path_node_img_.copy()
        path_node_temp = []
        draw_path_node_temp = []
        color_roud = 10 * roud
        for node_id_ in path_node_id:
            obj_id_get = self.matcher.get(node_id_)
            path_node_temp.append(obj_id_get["abs_xyz"])
        for draw_node_ in path_node_temp:
            if isinstance(draw_node_, str):
                # print("draw_node_ is:", draw_node_[1:-1])
                draw_node_ = list(draw_node_[1:-1].split(','))
                draw_node_ = [float(change_xyz) for change_xyz in draw_node_]
            [cv_x_get, cv_y_get, cv_x_end_, cv_y_end_] = self.real_2_cv_xyz(draw_node_[0], draw_node_[2])
            cv2.circle(path_node_img_, (cv_x_get, cv_y_get), 5, (2 + color_roud, 1 + color_roud, 2 + color_roud), 1)
            draw_path_node_temp.append([cv_x_get, cv_y_get])
        short_line = np.array(draw_path_node_temp)
        cv2.polylines(path_node_img_, [short_line], isClosed=False,
                      color=(2 + color_roud, 255 - color_roud, 2 + color_roud), thickness=1)
        cv2.imwrite(process_sub_epoch + "/path_node_img_" + str(roud) + ".jpg", path_node_img_)
        self.save_img_node_num += 1

        cv2.waitKey(1)

    # Get close object
    def get_near_obj(self, dis=50):
        nodes_id_get = get_obj_visible()
        near_nodes = []
        if self.obj_id in nodes_id_get:
            obj_id_xyz = obj2xyz(self.obj_id)
            for node_vis_ in nodes_id_get:
                vis_node_xyz = obj2xyz(node_vis_)
                obj_surrund_dis = np.sqrt(sum(np.power((vis_node_xyz - obj_id_xyz), 2)))
                if obj_surrund_dis < dis:
                    near_nodes.append(node_vis_)
            return near_nodes
        else:
            return [self.obj_id]

    # Similar objects are grouped in OBSERVE
    def save_observed_nodes(self, nodes_id_get):
        bot.OBSERVED_NODE.extend(nodes_id_get)

    # Navigate to the target point
    def navigate_2_target(self):  # record whether you go through a door, so as to record the state of crossing the room
        self.event = self.controller.step(action="Stand")
        obj_find_xyz = obj2xyz(obj_id=self.obj_id, show=False)

        r_end = pre_obj_size(self.obj_id)

        path, judge = self.guide(r_begin=radius, goal=[obj_find_xyz[0], obj_find_xyz[2] - 20], r_goal=r_end,
                                 mod=True, path_show=False)
        if judge:
            need_save_nodes_id = bot.get_near_obj(dis=500)
            navigate_true = bot.path_movement(path)
            # print("navigate_true:", navigate_true)
            if navigate_true:
                self.save_observed_nodes(need_save_nodes_id)
            else:
                # print("Navigation failed. Target point failed")
                pass

        else:
            # print("Maidens in prayer")
            count_w = 50
            right_point = (int(self.maps_robot_pos[0]), self.maps_robot_pos[1])
            left_point = (int(self.maps_robot_pos[0] - count_w), self.maps_robot_pos[1])
            left_obs_num = np.count_nonzero(
                self.maps_path_2d[left_point[1]:left_point[1] + count_w, left_point[0]:left_point[0] + count_w])
            right_obs_num = np.count_nonzero(
                self.maps_path_2d[right_point[1]:right_point[1] + count_w, right_point[0]:right_point[0] + count_w])

            if left_obs_num > right_obs_num:
                self.event = self.controller.step(action='RotateRight')
                posible = [00, 70]
                path, judge = bot.guide(r_begin=radius, goal=posible, r_goal=10,
                                        mod=False, path_show=False)
                if judge:
                    bot.path_movement(path)
                self.event = self.controller.step(action='RotateLeft')
            else:

                self.event = self.controller.step(action='RotateLeft')
                posible = [00, 50]
                path, judge = bot.guide(r_begin=radius, goal=posible, r_goal=10,
                                        mod=False, path_show=False)
                if judge:
                    bot.path_movement(path)
                self.event = self.controller.step(action='RotateRight')

    def check_current_env(self):
        if self.navigation_path_obj_id:
            self.obj_id = self.navigation_path_obj_id[0]
            return True
        else:
            return False

    # Detect current status
    def check_robot_state(self):
        now_vis_node = get_obj_visible()
        self.event = self.controller.step(action="Stand")
        self.objects = self.event.metadata['objects']
        self.agent = self.event.metadata['agent']

        diff_dis = np.linalg.norm(np.array([self.finally_xyz[0], self.finally_xyz[2]]) - np.array(
            [self.agent['position']['x'],
             self.agent['position']['z']])
                                  )
        if diff_dis <= 1 and self.finally_target in now_vis_node:
            print("\n Congratulattion！\n")
            bot.SUCCESS_ARRIVE = True


        elif bot.OBSERVED_NODE == []:
            bot.SUCCESS_ARRIVE = False
        elif self.obj_id in bot.OBSERVED_NODE:
            if self.finally_target == self.obj_id and self.finally_target in now_vis_node:
                warnings.warn("dead lock!")
                # print("The current distance is:", diff_dis)
                bot.SUCCESS_ARRIVE = True
            else:
                # print("Target is get now but dis is too far:", self.obj_id)
                bot.SUCCESS_ARRIVE = False
        else:
            bot.SUCCESS_ARRIVE = False

        if self.save_img_node_num > 15:
            if diff_dis <= 1:
                self.SUCCESS_ARRIVE = True
            else:
                self.SUCCESS_ARRIVE = False
        return bot.SUCCESS_ARRIVE

    # Bypassing larger, potentially obscuring obstacles
    def avoid_obstacles(self, avoid_obj):
        # print("To try to bypass this obstacle::", avoid_obj)
        count_w = 30
        bot.maps_path_2d, bot.maps_info = build_map()  # update local map
        obstacle_point = (int(self.maps_robot_pos[0] - count_w), self.maps_robot_pos[1])
        obstacle_point_area = np.count_nonzero(
            self.maps_path_2d[obstacle_point[1]:obstacle_point[1] + count_w,
            obstacle_point[0]:int(obstacle_point[0] + 2 * count_w)])
        avoid_test_img = bot.maps_path_2d.copy()
        cv2.circle(avoid_test_img, [self.maps_robot_pos[0], self.maps_robot_pos[1]], 4, (123, 12, 3), 1)
        cv2.rectangle(avoid_test_img, [obstacle_point[0], obstacle_point[1]],
                      [int(obstacle_point[0] + 2 * count_w), obstacle_point[1] + count_w], (23, 12, 323), 2)

        if obstacle_point_area > 1200:
            # print("Get out of the way and back up")
            self.event = self.controller.step(action="RotateLeft", degrees=180)
            path, judge = bot.guide(r_begin=radius, goal=[0, 140], r_goal=10,
                                    mod=False, path_show=False)
            if judge:
                self.path_movement(path)
            self.event = self.controller.step(action="RotateLeft", degrees=180)

        obj_get_in_avoid_ = get_obj_visible()
        if avoid_obj in obj_get_in_avoid_:
            # Find the other location of the previous target
            temp_size = pre_obj_size(avoid_obj)
            temp_obj_xyz = obj2xyz(avoid_obj)
            posible = [temp_obj_xyz[0] - temp_size, temp_obj_xyz[2] + 30]
            path, judge = bot.guide(r_begin=radius, goal=posible, r_goal=temp_size,
                                    mod=True, path_show=False)
            if judge:
                self.path_movement(path)
            else:
                posible = [temp_obj_xyz[0] + temp_size, temp_obj_xyz[2] + 30]
                path, judge = bot.guide(r_begin=radius, goal=posible, r_goal=temp_size,
                                        mod=False, path_show=False)
                if judge:
                    self.path_movement(path)
                # else:
                #     print("There's still no proper path to explore")
        # else:
        #     print("Turn around and the target is gone")

    # Find the shortest path
    def find_shortest_path_nodes(self):
        self.navigation_path_obj_id = []
        min_cost = float('Inf')
        short_turn_get = 0
        for turn_i in range(4):
            nodes_id_get = get_obj_visible()
            create_relationships(nodes_id_get)
            if self.finally_target in get_obj_visible():
                self.navigation_path_node = [
                    self.gds.find_node_id(["object"], {"obj_id": self.finally_target})]
                self.navigation_path_obj_id = [self.finally_target]
                short_turn_get = 0
                min_cost = 0
                break

            bot.G = update_gds_graph(gds_get=bot.gds, G_name="navigation_graph")
            for nodes_vis_id_ in nodes_id_get:
                get_short_temp = []
                shortest_path_get, path_cost = get_shortest_path(gds_get=self.gds, G_get=self.G, star_id=nodes_vis_id_,
                                                                 end_id=bot.finally_target)
                # Choose the path with the shortest cost
                if shortest_path_get != None and min_cost > path_cost:
                    min_cost = path_cost
                    self.navigation_path_node = shortest_path_get
                    # print("shortest_path_get:", shortest_path_get)
                    short_turn_get = turn_i

                    for path_node_id in self.navigation_path_node:
                        obj_id_get = self.matcher.get(path_node_id)
                        get_short_temp.append(obj_id_get["obj_id"])
                    self.navigation_path_obj_id = get_short_temp

            self.event = self.controller.step(action="RotateLeft")

        # Judge the test result
        if math.isinf(min_cost):
            # print("If the shortest path is not found, it will be searched randomly according to probability")
            return False
        else:
            back_rotate = int(short_turn_get * 90)
            self.event = self.controller.step(action="RotateLeft", degrees=back_rotate)
            temp_obj = self.navigation_path_obj_id[0]

            if (len(bot.OBSERVED_NODE) > 2 and
                    bot.OBSERVED_NODE[-1] == bot.OBSERVED_NODE[-2] == temp_obj):

                # Try to bypass the obstruction and head for another direction
                self.avoid_obstacles(temp_obj)
                # Determine if the step limit has been exceeded
                if bot.ALL_STEP > max_step:
                    # print("If the maximum number of steps is exceeded, no further action is required")
                    return "over_step"
                else:
                    shortest_flag = self.find_shortest_path_nodes()
                if shortest_flag == "over_step":
                    return False

            return True

    # Follow the path to navigate
    def navi_with_path(self):
        self.success_flag = self.check_current_env()
        # print("Ready to go to the node：", self.obj_id, self.success_flag)
        if self.success_flag:
            self.navigate_2_target()
            self.check_robot_state()

        if self.obj_id.split(DELIMITER)[0] in bot.SPECIAL_NODE:
            now_id = get_obj_visible()
            if self.obj_id in now_id:
                # bot.SPECIAL_OBSERVED.append(self.obj_id)
                indoor_flag = self.into_door(self.obj_id)
                if not indoor_flag:
                    self.avoid_obstacles(self.obj_id)

    # Enter the room through the door
    def into_door(self, door_id):
        obstacle_door_area_ = 1.1
        obstacle_temp_ = 1.0
        door_turn_ = 0
        width_ = 15
        height_ = 100
        for in_door_turn_ in range(4):
            in_door_vis = get_obj_visible()
            if door_id in in_door_vis:
                maps_path_2d, door_maps_info = build_map()
                obj_real_xyz = obj2xyz(door_id)
                door_xyz = [(obj_real_xyz[0] - door_maps_info[0]).astype(int),
                            (obj_real_xyz[2] - door_maps_info[2] - 20).astype(int)]
                cv2.circle(maps_path_2d, door_xyz, 3, (232, 123, 12), 1)
                cv2.rectangle(maps_path_2d, (door_xyz[0] - width_, 0), (door_xyz[0] + width_, door_xyz[1] + height_),
                              (123, 23, 223), 1)

                door_area_ = maps_path_2d[0:door_xyz[1] + height_,
                             door_xyz[0] - width_:door_xyz[0] + width_]
                obstacle_get = np.count_nonzero(door_area_)
                all_get = door_area_.shape[0] * door_area_.shape[1] + 1
                obstacle_temp_ = obstacle_get / all_get  # Calculation of obstacle ratios
                if obstacle_door_area_ > obstacle_temp_ and door_maps_info[3] > 770:
                    obstacle_door_area_ = obstacle_temp_
                    door_turn_ = in_door_turn_
            self.event = self.controller.step(action="RotateLeft")
        if door_turn_ == 2:
            return True
        all_turn_ = int(90 * door_turn_)
        self.event = self.controller.step(action="RotateLeft", degrees=all_turn_)
        # print("obstacle_get/all_get:", obstacle_door_area_, door_id)
        if obstacle_door_area_ < 0.55:
            path, judge = self.guide(r_begin=radius, goal=[0, 300], r_goal=10,
                                     mod=False, path_show=False)
            if judge:
                self.path_movement(path)
                bot.SPECIAL_OBSERVED.append(door_id)
                # self.cross_observed.append(door_id)
            # bot.SPECIAL_OBSERVED.append(door_id)
            return True
        else:
            return False

    # Path judgement by probability when no path is found
    def find_possible_nodes(self, find_rate=0.1):
        possible_turn_get = 0
        # special_turn_get = 0
        possible_obj_id = None
        # limit_z = 10000
        for turn_i_possible_ in range(4):
            limit_z_all = [3000]
            nodes_id_get = get_obj_visible()
            create_relationships(nodes_id_get)
            max_rate = 0
            try:
                target_ekb_index = self.obj_names.index(self.finally_type)
            except ValueError as e:
                print("find_possible_nodes function error:", e)
                # print("The type is:", self.finally_type)
            for vis_id_get_ in nodes_id_get:
                vis_name_get_ = vis_id_get_.split(DELIMITER)[0]
                if vis_name_get_ in bot.SPECIAL_NODE:
                    obj_real_xyz = obj2xyz(vis_id_get_)
                    limit_z_all.append(obj_real_xyz[2])
                limit_z = np.min(limit_z_all)
                # print("limit_z:", limit_z)
            for vis_id_get_ in nodes_id_get:
                vis_name_get_ = vis_id_get_.split(DELIMITER)[0]
                try:
                    name_id = self.obj_names.index(vis_name_get_)
                except ValueError as e:
                    continue
                possible_rate = self.obj_rates[target_ekb_index][name_id]  # correlation of proximity to the final target
                # print("possible_rate:", possible_rate, vis_name_get_)
                if bot.SEARCH_MODE:  # Activate global search mode
                    possible_rate = 0.5
                get_obj_real_xyz = obj2xyz(vis_id_get_)
                # print("The possible_rate is:", possible_rate, vis_name_get_, get_obj_real_xyz)
                if possible_rate >= find_rate \
                        and vis_name_get_ not in bot.NO_DETECTION \
                        and vis_name_get_ not in bot.SPECIAL_NODE \
                        and vis_id_get_ not in bot.OBSERVED_NODE \
                        and possible_rate > max_rate \
                        and get_obj_real_xyz[2] < limit_z:
                    max_rate = possible_rate
                    possible_turn_get = turn_i_possible_
                    possible_obj_id = vis_id_get_

            self.event = self.controller.step(action="RotateLeft")

        back_rotate = int(possible_turn_get * 90)

        self.event = self.controller.step(action="RotateLeft", degrees=back_rotate)
        # print(f"The possible_obj_id is:{possible_obj_id}")
        if possible_obj_id is None:
            # print("There's nothing left to find.")
            self.door_search()

        elif possible_obj_id is not None:
            self.obj_id = possible_obj_id
            self.navigate_2_target()
            self.check_current_env()

        if possible_obj_id == None:
            if find_rate <= 0.021:
                # print("A forced movement will be performed to break the deadlock")
                self.check_robot_state()

                for compultion_move_ in range(4):
                    self.event = bot.controller.step(action="RotateLeft")

                    posible = [0, 210]
                    path, judge = bot.guide(r_begin=radius, goal=posible, r_goal=10,
                                            mod=False, path_show=False)
                    if judge:
                        self.path_movement(path)
                        break
                    else:
                        continue

    # Calculation of penalty weights given to the gate and other nodes
    def node_penalty_weights(self, node_id):
        weights_get = 0
        if node_id in bot.SPECIAL_OBSERVED:
            weights_get = bot.SPECIAL_OBSERVED.index(node_id) + bot.SPECIAL_OBSERVED.count(node_id) * 2
        return weights_get

    # Explore by key nodes such as doors for global searches
    def door_search(self):
        # special_turn_get = 0
        possible_obj_id = None
        min_penalty_value_ = float('inf')
        turn_i_possible_get_ = 0
        for turn_i_possible_ in range(4):
            nodes_id_get = get_obj_visible()
            # print("nodes_id_get now is:",nodes_id_get)
            create_relationships(nodes_id_get)

            for vis_id_get_ in nodes_id_get:
                vis_name_get_ = vis_id_get_.split(DELIMITER)[0]
                # if vis_name_get_ in bot.SPECIAL_NODE and vis_id_get_ not in bot.SPECIAL_OBSERVED:
                # print("vis_id_get_:",vis_id_get_,nodes_id_get)
                if vis_name_get_ in bot.SPECIAL_NODE:
                    # print("Identify special target points for special care:", vis_name_get_, vis_id_get_)
                    node_penalty_value_ = self.node_penalty_weights(vis_id_get_)
                    # print("penalty_value_", node_penalty_value_)
                    if min_penalty_value_ > node_penalty_value_:
                        min_penalty_value_ = node_penalty_value_
                        possible_obj_id = vis_id_get_
                        turn_i_possible_get_ = turn_i_possible_
                        # break

            self.event = self.controller.step(action="RotateLeft")

        back_rotate = int(turn_i_possible_get_ * 90)
        self.event = self.controller.step(action="RotateLeft", degrees=back_rotate)

        # print(f"The possible_obj_id is:{possible_obj_id}")
        # if possible_obj_id is not None and possible_obj_id != bot.SPECIAL_OBSERVED[-1]:
        if possible_obj_id is not None:
            self.obj_id = possible_obj_id
            # bot.OBSERVED_NODE.append(self.obj_id)
            self.navigate_2_target()

            nodes_now_id = get_obj_visible()
            indoor_flag = None
            bot.SPECIAL_OBSERVED.append(possible_obj_id)

            if possible_obj_id in nodes_now_id:
                indoor_flag = self.into_door(possible_obj_id)
            else:
                pass
            if indoor_flag is False:
                self.avoid_obstacles(possible_obj_id)
                indoor_again = self.into_door(possible_obj_id)
        else:
            area_turn_ = 0
            area_info_temp_ = [0, 0, 0, 0]
            max_free_area = 0
            for compultion_move_ in range(4):
                map_door_get, map_info_ = build_map()
                # door_obstacle_area = np.count_nonzero(map_door_get)
                door_free_area = np.count_nonzero(map_door_get == 0)
                if door_free_area > max_free_area:
                    max_free_area = door_free_area
                    area_turn_ = compultion_move_
                    area_info_temp_ = map_info_
                # print("door_obstacle_area:", door_obstacle_area)
                # print("compultion_move:", compultion_move_, door_free_area, map_info_)
                # cv2.imshow("map_door_get", map_door_get)
                # cv2.waitKey(1)
                self.event = bot.controller.step(action="RotateLeft")
            avoid_all_turn_ = int(90 * area_turn_)
            self.event = bot.controller.step(action="RotateLeft", degrees=avoid_all_turn_)
            force_point = [0, int(area_info_temp_[3] / 2)]
            path, judge = bot.guide(r_begin=radius, goal=force_point, r_goal=10,
                                    mod=False, path_show=False)

            if judge:
                self.path_movement(path)
            # else:
            #     print("There's no way. One last shot")

    def draw_the_obj_point(self, obj):
        obj_img = get_third_img()
        # obj_real_xy = obj["abs_xyz"]
        [cv_x_get, cv_y_get, cv_x_end_, cv_y_end_] = self.real_2_cv_xyz(self.finally_xyz[0], self.finally_xyz[2])
        cv2.circle(obj_img, (cv_x_get, cv_y_get), 5, (180, 105, 255), 1)
        cv2.imshow("obj_img_show", obj_img)
        cv2.waitKey(0)

    # Keyboards Control
    def keyboards(self):
        print("Keyboard Control")
        while True:
            # x = input()
            # cv2.imshow("img_zero", img_zero)
            get_third_img()
            img_zero = np.zeros((240, 320))
            cv2.imshow("img_zero", img_zero)
            x = cv2.waitKey(0)
            if x == ord('w'):
                self.event = self.controller.step(dict(action='MoveAhead'))
                # self.agent = self.event.metadata['agent']
                # print(int(self.agent['position']['x'] * 100), int(self.agent['position']['y'] * 100),
                #       int(self.agent['position']['z'] * 100))
            if x == ord('s'):
                self.event = self.controller.step(dict(action='MoveBack'))
                # self.agent = self.event.metadata['agent']
                # print(int(self.agent['position']['x'] * 100), int(self.agent['position']['y'] * 100),
                #       int(self.agent['position']['z'] * 100))
            if x == ord('a'):
                self.event = self.controller.step(dict(action='MoveLeft'))
                # self.agent = self.event.metadata['agent']
                # print(int(self.agent['position']['x'] * 100), int(self.agent['position']['y'] * 100),
                #       int(self.agent['position']['z'] * 100))
            if x == ord('d'):
                self.event = self.controller.step(dict(action='MoveRight'))
                # self.agent = self.event.metadata['agent']
                # print(int(self.agent['position']['x'] * 100), int(self.agent['position']['y'] * 100),
                #       int(self.agent['position']['z'] * 100))
            if x == ord('q'):
                self.event = self.controller.step(action="RotateLeft", degrees=90)
                # self.event = self.controller.step(dict(action='RotateLeft',degrees=180))
                # self.agent = self.event.metadata['agent']
                self.theta += 90
            if x == ord('e'):
                self.event = self.controller.step(dict(action='RotateRight'))
                self.theta -= 90
            if x == ord('j'):
                self.navi_with_path()
            if x == ord('b'):
                self.check_robot_state()
            if x == ord('f'):
                bot.maps_path_2d, bot.maps_info = build_map()
                cv2.imshow("Find_map", bot.maps_path_2d)
                cv2.waitKey(1)
                nodes_id = get_obj_visible()
                print("nodes_id get to show:", nodes_id)
                create_relationships(nodes_id)
            elif x == ord('i'):
                first_path_node = self.matcher.get(bot.navigation_path_node[0])
                bot.obj_id = first_path_node["obj_id"]
                bot.obj_xyz = first_path_node["abs_xyz"]
            elif x == ord('l'):
                nodes_id = get_obj_visible()
                print("nodes_id:", nodes_id)
            elif x == ord('t'):
                print("Test now left")
                self.event = self.controller.step(action="RotateLeft", degrees=0)
            elif x == 27:
                break
            elif x == ord('t'):
                maps_path_2d, door_maps_info = build_map()
                nodes_id = get_obj_visible()
                for door_find_ in nodes_id:
                    if door_find_.split(DELIMITER)[0] in bot.SPECIAL_NODE:
                        obj_real_xyz = obj2xyz(door_find_)
                        door_xyz = [(obj_real_xyz[0] - door_maps_info[0]).astype(int),
                                    (obj_real_xyz[2] - door_maps_info[2] - 20).astype(int)]
                        cv2.circle(maps_path_2d, door_xyz, 3, (232, 123, 12), 1)
                        cv2.rectangle(maps_path_2d, (door_xyz[0] - 50, 0), (door_xyz[0] + 50, door_xyz[1] + 100),
                                      (123, 23, 223), 1)
                        door_area_ = maps_path_2d[0:door_xyz[1] + 100,
                                     door_xyz[0] - radius:door_xyz[0] + radius]
                        obstacle_get = np.count_nonzero(door_area_)
                        free_get = np.count_nonzero(door_area_ == 0)
                        all_get = radius * 2 * (door_xyz[1] + 100)
                        # print(f"door obj id is:{door_find_}"
                        #       f"obstacle_get:{obstacle_get},free_get{free_get},\n"
                        #       f"percentage of obstacle{obstacle_get / all_get},free of obstacle{free_get / all_get}")
                        cv2.imshow("door_area_", door_area_)
                        cv2.imshow("door_through_map", maps_path_2d)
                        cv2.waitKey(1)


def obj2xyz(obj_id, show=False):
    nowevent = bot.event
    obj_box_xyxy = nowevent.instance_detections2D[obj_id]
    dep_get = nowevent.depth_frame
    # if obj_id in ["door|1|3","door|2|3","door|3|3"]:
    #     show = True
    if obj_id[:4] == "door":
        # show = False
        # print("The door obj_box_xyxy is:", obj_box_xyxy)
        x_get = int((obj_box_xyxy[0] + obj_box_xyxy[2]) / 2)
        y_get = int(obj_box_xyxy[3] * 0.9)

        # if y_get > 600:
        #     y_get = int(obj_box_xyxy[3])
    else:
        x_get = int((obj_box_xyxy[0] + obj_box_xyxy[2]) / 2)
        y_get = int((obj_box_xyxy[1] + obj_box_xyxy[3]) / 2)
    if show:
        img_get = nowevent.cv2img
        img_get = img_get.copy()
        cv2.rectangle(img_get, (obj_box_xyxy[0], obj_box_xyxy[1]), (obj_box_xyxy[2], obj_box_xyxy[3]), (23, 23, 123), 3)
        cv2.circle(img_get, [x_get, y_get], 5, (234, 12, 32, 2))
        cv2.imshow("img_get_obj", img_get)
        cv2.waitKey(0)

    depth_xyz = depth2xyz(x_get, y_get, dep_get)
    return depth_xyz * amplify


# Query the corresponding obj_id (id in ai2thor, i.e., the object's id in the environment) via Cypher and return the corresponding attribute values
def get_scense_id(property_key):
    results = []
    cypher_query = f"Match (n) return n.{property_key} as property"
    results_item = bot.map_graph.run(cypher_query).data()
    for r_item in results_item:
        results.append(r_item["property"])
    return results

# 3d to 2d
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
    map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel, iterations=3)
    kernel_dilate = np.ones((grid_size * 2, grid_size * 2), np.uint8)
    map = cv2.dilate(map, kernel_dilate, 2)
    kernel = np.ones((int(radius * 1.8), int(radius * 1.8)), np.uint8)
    # map is real map and maps is navigation maps
    maps = cv2.dilate(map, kernel)
    return maps


def build_map(event_get=None, show=False):
    if event_get:
        event = event_get
    else:
        event = bot.controller.step(action="Stand")
    width, height = 600, 600
    fov = 90
    height_min = -0.12
    height_max = 1.5
    # Convert fov to focal length
    focal_length = 0.5 * height / np.tan(to_rad(fov / 2))

    # camera intrinsics
    fx, fy, cx, cy = (focal_length, focal_length, width / 2, height / 2)

    # Obtain point cloud
    color = o3d.geometry.Image(event.frame.astype(np.uint8))
    dep_ori = event.depth_frame
    dep = dep_ori / 1.0  # If not dvision, something will be wrong??!
    depth = o3d.geometry.Image(dep)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                              depth_scale=1.0,
                                                              depth_trunc=10,
                                                              convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # R = rotate_mat_all([0, 0, 0])
    nb_neighbors = 30
    std_ratio = 2.0
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                              std_ratio=std_ratio)
    rx = -np.pi / 4
    ry = 0
    rz = 0
    R = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    pcd.rotate(R, (0, 0, 0))
    map_vtx = np.asarray(pcd.points)
    npy_judge = np.where(
        (map_vtx[:, 1] > height_min) & (
                map_vtx[:, 1] < height_max))
    map_change = map_vtx[npy_judge]

    map_ground = np.where(map_vtx[:, 1] > height_max)
    ground_point = map_vtx[map_ground]

    map_x, map_y = (map_vtx[:, 0] * amplify).astype(int), (
            map_vtx[:, 2] * amplify).astype(int)
    map_info_ori = [np.min(map_x), np.max(map_x), np.min(map_y), np.max(map_y)]
    ground_2d = three2tow(ground_point[:, [0, 2]], map_info_ori)
    ground_2d = cv2.bitwise_not(ground_2d)
    tri_x0 = int(0 - map_info_ori[0])
    tri_y1 = int(map_info_ori[2]) + 50
    # print("tri_y1:",tri_y1)
    # tri_y1 = 101.00
    triangle_cnt = np.array([(tri_x0, 0), (tri_x0 + tri_y1, tri_y1), (tri_x0 - tri_y1, tri_y1)])
    # cv2.imshow("ground_2d_no_triangle", ground_2d)
    cv2.drawContours(ground_2d, [triangle_cnt], 0, (0, 0, 0), -1)

    obstacle_2d = three2tow(map_change[:, [0, 2]], map_info_ori)

    path_2d = ground_2d + obstacle_2d

    bot.maps_robot_pos = (tri_x0, 0)
    # cv2.imshow("ground_2d", ground_2d)
    # cv2.imshow("path_2d", path_2d)
    # cv2.imshow("obstacle_2d", obstacle_2d)
    # For test the 3D vector
    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.10, resolution=100)
    # mesh_center = [-0.6488081924120583, 0.5511720081170399, 0.9448663] # From lsy depth
    # mesh_center = [-0.6861704051494597, 1.1103484737873075, 1.2475826]  # From dbc absolute depth
    # mesh_center = [-1.2852787971496582, -0.49742767214775085, -0.6937699913978577]  # From dbc absolute depth
    # new one? -1.2241117656230927,0.1343778371810913,-0.7132988572120667
    # TVStand -1.0402389168739319,-0.9014184474945068,-0.7722395062446594]
    # Floor -0.75,-0.9009991884231567,0.0
    # mesh_sphere.translate(mesh_center)
    # mesh_sphere.compute_vertex_normals()
    if show:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, axis_pcd])

        pcd_obv = o3d.geometry.PointCloud()
        pcd_obv.points = o3d.utility.Vector3dVector(map_change)
        o3d.visualization.draw_geometries([pcd_obv, axis_pcd])

    map_info_change = map_info_ori
    map_info_change[2] = 0
    # print(map_vtx)
    # return map_change
    return path_2d, map_info_change

# Calculate the number of obstacles between two points
def line_scan(image_ori, start, end, r_start=10, r_end=10):
    image = image_ori.copy()
    mask = np.zeros_like(image)
    cv2.line(mask, start, end, 255, bot.LINE_WIDTH)
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


def init_draw(position_start, frame):
    img_draw = get_third_img()
    img_draw = img_draw.copy()
    cv2.circle(img_draw, (position_start[0], position_start[1]), 5, (234, 12, 32), 1)
    cv2.putText(img_draw, '0', ((position_start[2] - 20, position_start[3] + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (12, 232, 123), 2)
    cv2.imwrite(process_sub_epoch + "/real" + 'start' + "experiment.jpg", img_draw)
    cv2.imwrite(process_sub_epoch + "/observe" + 'start' + ".jpg", frame)


# Draw the progress
def draw_point(number=None):
    img_draw = get_third_img()
    img_draw = img_draw.copy()
    move_line = []
    if number == None:
        for draw_i in range(len(bot.RECORD_NODE)):
            # cv_x = 30 * bot.agent['position']['x'] + 65
            # cv_y = -26.7 * bot.agent['position']['z'] + 550
            # cv_x_get, cv_y_get = int(cv_x), int(cv_y)
            # rot_get = bot.agent['rotation']
            move_line.append((bot.RECORD_NODE[draw_i][0], bot.RECORD_NODE[draw_i][1]))
            cv2.circle(img_draw, (bot.RECORD_NODE[draw_i][0], bot.RECORD_NODE[draw_i][1]), 5, (234, 12, 32), 1)
            cv2.arrowedLine(img_draw, (bot.RECORD_NODE[draw_i][0], bot.RECORD_NODE[draw_i][1]),
                            (bot.RECORD_NODE[draw_i][2], bot.RECORD_NODE[draw_i][3]), (21, 246, 10), 2, 9, 0, 0.3)
            cv2.putText(img_draw, str(draw_i + 1), ((bot.RECORD_NODE[draw_i][2] - 20, bot.RECORD_NODE[draw_i][3] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (12, 232, 123), 2)
        move_line = np.array(move_line)
        cv2.polylines(img_draw, [move_line], isClosed=False, color=(223, 22, 24), thickness=3)
    else:
        move_line = [(node[0], node[1]) for node in bot.RECORD_NODE]
        move_line = np.array(move_line)
        cv2.polylines(img_draw, [move_line], isClosed=False, color=(223, 22, 24), thickness=3)
        for i in range(1, 8):
            cv2.circle(img_draw, (bot.RECORD_NODE[number - 1][0], bot.RECORD_NODE[number - 1][1]), i, (0, 0, 255), 1)
        cv2.putText(img_draw, str(number), ((bot.RECORD_NODE[number - 1][2] - 20, bot.RECORD_NODE[number - 1][3] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (12, 232, 123), 2)
    if number == None:
        cv2.imwrite(process_sub_epoch + "/real.jpg", img_draw)
    else:
        cv2.imwrite(process_sub_epoch + "/real" + str(number) + "experiment.jpg", img_draw)
    # cv2.waitKey(1)


# Setting up the third view camera
def get_top_down_frame():
    third_event = bot.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(third_event.metadata["actionReturn"])
    bounds = third_event.metadata["sceneBounds"]["size"]
    max_bounds = max(bounds["x"], bounds["z"])
    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bounds
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    third_event = bot.controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = third_event.third_party_camera_frames[-1]
    return top_down_frame


# Update the third
def get_third_img():
    third_event = bot.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(third_event.metadata["actionReturn"])
    bounds = third_event.metadata["sceneBounds"]["size"]
    max_bounds = max(bounds["x"], bounds["z"])
    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bounds  # 17.92
    # pose["position"]["y"] = 20
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]
    third_event = bot.controller.step(
        action="UpdateThirdPartyCamera",
        # thirdPartyCameraId=0,
        # # position=dict(x=-0, y=3.8, z=0),
        # position=dict(x=9.92, y=3.71, z=2.36),
        # rotation=dict(x=90, y=0, z=0),
        # fieldOfView=120
        **pose,
    )
    bot.event = bot.controller.step(action="Stand")
    bot.agent = bot.event.metadata['agent']

    third_img_get = third_event.third_party_camera_frames
    third_img_get = third_img_get[0].copy()
    third_img_get = cv2.cvtColor(third_img_get, cv2.COLOR_RGB2BGR)
    # cv2.imshow("third_img", third_img_get)
    # cv2.waitKey(1)
    return third_img_get


# neo4j gds upadate
def update_gds_graph(gds_get, G_name):
    G_exist = gds_get.graph.exists(G_name)["exists"]
    if G_exist:
        G_get = gds_get.graph.get(G_name)
        gds_get.graph.drop(G_get)
    G, project_result = gds_get.graph.project(G_name, "object", {
        bot.RELATIONSHIP_NAME: {"properties": ["composite_path_cost"]}})

    return G


# neo4j gds find the shortest path
def get_shortest_path(gds_get, G_get, star_id, end_id):
    end_exists = bot.matcher.match("object", obj_id=end_id).first()
    if end_exists != None:
        # Task 1 Object navigation
        # node_1 = matcher.match("object", Type=star_id).first().identity
        # node_2 = matcher.match("object", Type=end_id).first().identity
        # Task 2 Image Navigation
        node_1 = gds_get.find_node_id(["object"], {"obj_id": star_id})
        node_2 = gds_get.find_node_id(["object"], {"obj_id": end_id})
        result = gds_get.shortestPath.dijkstra.stream(
            G_get,
            sourceNode=node_1,
            targetNode=node_2,
            relationshipWeightProperty="composite_path_cost"
        )
        if result.empty:
            return None, False
        result_id = result['nodeIds'].values[0]
        path_cost = result["totalCost"][0]
        return result_id, path_cost
    else:
        return None, False


# Get the object in current image
def get_obj_visible():
    now_node = []
    bot.event = bot.controller.step(action="Stand")
    object_data = bot.event.metadata['objects']

    for obj in object_data:
        if obj["visible"] and obj["objectType"] not in bot.NO_DETECTION:
            try:
                xyxy_judge = bot.event.instance_detections2D[obj["objectId"]]
                now_node.append(obj["objectId"])
                if obj["objectId"] not in bot.NODE_EXISTED:
                    bot.NODE_EXISTED.append(obj["objectId"])
                    abs_xyz = [obj['position']['x'], obj['position']['y'], obj['position']['z']]
                    node = Node('object', obj_id=obj["objectId"], Type=obj['objectType'], abs_xyz=abs_xyz)
                    bot.map_graph.create(node)
            except KeyError:
                pass
    # print("node list:",now_node)
    return now_node


# Getting an estimate of the likely size of an object
def pre_obj_size(obj_get):
    obj_name = obj_get.split(DELIMITER)[0]
    try:
        radius_id = bot.obj_names.index(obj_name)
        obj_size = int(bot.obj_avg_radius[radius_id] * amplify / 3)
    except:
        obj_size = 15
    return obj_size


# Creating Map Relationships
def create_relationships(nodes_get):
    path_all = []
    cost_all = []
    r_st_get = []
    r_en_get = []
    star_name_get = []
    end_name_get = []
    bot.maps_path_2d, bot.maps_info = build_map()
    map_obj_judge, map_info_judge = bot.maps_path_2d, bot.maps_info
    # for node_id_get_first in tqdm(range(len(nodes_get)), desc="Create relationships now"):
    for node_id_get_first in range(len(nodes_get)):
        node_1 = bot.matcher.match("object", obj_id=nodes_get[node_id_get_first]).first()
        for node_id_get_second in range(node_id_get_first + 1, len(nodes_get)):
            # print("node_id_get to create:", node_id_get_first, node_id_get_second)
            # Get xyz from u,v
            obj_xyz_first = obj2xyz(obj_id=nodes_get[node_id_get_first])
            obj_xyz_second = obj2xyz(obj_id=nodes_get[node_id_get_second])
            obj_xyz_first_2d = [(obj_xyz_first[0] - map_info_judge[0]).astype(int),
                                (obj_xyz_first[2] - map_info_judge[2] - 20).astype(int)]
            obj_xyz_second_2d = [(obj_xyz_second[0] - map_info_judge[0]).astype(int),
                                 (obj_xyz_second[2] - map_info_judge[2] - 20).astype(int)]
            r_start = pre_obj_size(nodes_get[node_id_get_first])
            r_end = pre_obj_size(nodes_get[node_id_get_second])
            pixel_count_get = line_scan(map_obj_judge, obj_xyz_first_2d,
                                        obj_xyz_second_2d, r_start=r_start,
                                        r_end=r_end)  # pixel_count_get Reflects to some extent the number and complexity of obstacles between two objects
            # print("pixel_count_get:",pixel_count_get)
            # print("pixel_count_get:",pixel_count_get)
            if pixel_count_get < 490:  # 280 # 490:
                path_direct = [obj_xyz_first_2d, obj_xyz_second_2d]
                path_all.append(path_direct)
                cost_all.append(pixel_count_get)
                r_st_get.append(r_start)
                r_en_get.append(r_end)

                node_2 = bot.matcher.match("object", obj_id=nodes_get[node_id_get_second]).first()

                star_name_get.append(node_1['Type'])
                end_name_get.append(node_2['Type'])

                euclidean_dis = int(np.linalg.norm(np.array(obj_xyz_first_2d) - np.array(obj_xyz_second_2d)))
                # euclidean_dis = int(euclidean_dis)
                composite_path_cost = int(
                    euclidean_dis * (1 - bot.OBS_PARA) + pixel_count_get * bot.OBS_PARA)
                rel_1 = Relationship(node_1, bot.RELATIONSHIP_NAME, node_2, obstacle_path=pixel_count_get,
                                     euclidean_dis=euclidean_dis, composite_path_cost=composite_path_cost)
                rel_2 = Relationship(node_2, bot.RELATIONSHIP_NAME, node_1, obstacle_path=pixel_count_get,
                                     euclidean_dis=euclidean_dis, composite_path_cost=composite_path_cost)
                bot.map_graph.create(rel_1)
                bot.map_graph.create(rel_2)
                # print("path_cost:",path_cost)
    for dir_id in range(len(path_all)):
        test_map_img = cv2.merge([map_obj_judge] * 3)
        cv2.circle(test_map_img, path_all[dir_id][0], r_st_get[dir_id], (0, 0, 0), -1)
        cv2.circle(test_map_img, path_all[dir_id][1], r_en_get[dir_id], (0, 0, 0), -1)
        cv2.circle(test_map_img, path_all[dir_id][0], 5, (241, 23, 0), 2)
        cv2.circle(test_map_img, path_all[dir_id][1], 5, (0, 23, 241), 2)
        cv2.line(test_map_img, path_all[dir_id][0], path_all[dir_id][1], (20, 230, 10), bot.LINE_WIDTH)


def get_rotate(theta):
    return np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_mat_all(angles):
    R = o3d.geometry.get_rotation_matrix_from_yzx(angles).T
    return R


def to_rad(th):
    return th * np.pi / 180


def depth2xyz(u, v, depth):
    # Convert fov to focal length
    fy = 0.5 * height_set / np.tan(to_rad(fov / 2))
    fx = 0.5 * width_set / np.tan(to_rad(fov / 2))
    fx, fy, cx, cy = (fx, fy, width_set / 2, height_set / 2)
    z_real = depth[v][u]
    x_real = (u - cx) * z_real / fx
    y_real = (v - cy) * z_real / fy
    return np.array([x_real, y_real, z_real])


# Random choose the targets
def get_target_id_assemble():
    all_obj = bot.event.metadata["objects"]
    end_range = len(all_obj)
    # num_sample = 10
    num_sample = max(20, int(len(all_obj) / 8))
    if num_sample > end_range:
        print("The amount of data is too small to meet the random "
              "sample size requirement and the number of samples will be reduced.",
            end_range)
        num_sample = end_range
    random.seed(1)
    random_num_get = random.sample(range(0, end_range), num_sample)
    target_id_choose = []

    for obj_num in random_num_get:
        random_obj = all_obj[obj_num]
        if random_obj['name'].split(DELIMITER)[0] in bot.obj_names:
            target_id_choose.append(random_obj['objectId'])

        else:
            continue
        if len(target_id_choose) >= 20:
            break
    return target_id_choose


# Determine whether a target exists and is appropriate
def check_target():
    all_obj = bot.event.metadata["objects"]
    target_exist = False
    for every_obj in all_obj:
        if every_obj['objectId'] == bot.finally_target:
            bot.finally_type = bot.finally_target.split(DELIMITER)[0]
            bot.ai2_final_xyz = every_obj["position"]
            bot.ai2_final_rot = every_obj["rotation"]
            bot.ai2_final_xyz['y'] = 0
            bot.finally_xyz = [bot.ai2_final_xyz['x'], bot.ai2_final_xyz['y'], bot.ai2_final_xyz['z']]
            try:
                bot.shortest_node_get = get_shortest_path_to_object(
                    controller=bot.controller,
                    object_id=bot.finally_target,
                    initial_position=bot.event.metadata["agent"]["position"]
                )
            except ValueError as e:
                return False
            bot.shortest_node_get.append(bot.ai2_final_xyz)
            target_exist = True
            bot.the_shortest_path_draw(bot.shortest_node_get)
            bot.shortest_path_value = path_distance(bot.shortest_node_get)
            # print("bot.shortest_path_value:",bot.shortest_path_value)
            return True
            # break

    assert target_exist, "Object is not exist'{}'".format(bot.finally_target)


# Visual Navigation Search
def search_process():
    bot.collect_now_node()
    if random_choose:
        random_of_control()  # Random walking
    else:
        for create_surround_ in range(4):
            # View the target in the current field of view
            nodes_id = get_obj_visible()
            create_relationships(nodes_id)
            # print("maps get and info get",bot.maps_path_2d, bot.maps_info)
            bot.event = bot.controller.step(action="RotateLeft")

        for explore_i in range(60):
            bot.check_robot_state()
            print("bot.SUCCESS_ARRIVE:", bot.SUCCESS_ARRIVE)
            if bot.SUCCESS_ARRIVE:
                print("Task OVER, Action STOP NOW")
                bot.controller.step(action="Stand")
                # bot.controller.stop()
                bot.save_img_node_num = 0
                break
            bot.G = update_gds_graph(gds_get=bot.gds, G_name="navigation_graph")
            find_shortest_flag = bot.find_shortest_path_nodes()

            if bot.SEARCH_MODE:
                bot.door_search()
            else:
                if find_shortest_flag:
                    bot.draw_path_node(bot.navigation_path_node, explore_i)
                    target_get = bot.navi_with_path()
                else:
                    bot.find_possible_nodes()
        check_flag = bot.check_robot_state()

        print(f"The {explore_i + 1}th exploration is over.")
        bot.ikb_update_ekb()
        # print("Node data get:", bot.OBSERVED_NODE)
        draw_point()
        cv2.waitKey(1)


# random walking
def random_of_control():
    logger.info('Random Walking.')
    move_action = ["RotateLeft", "RotateLeft", "MoveLeft", "MoveRight", "MoveAhead"]
    num_step = 10
    for i in range(num_step):
        actions = random.choice(move_action)
        bot.controller.step(action=actions)
        bot.collect_now_node()
        time.sleep(0.005)
    bot.ALL_STEP += num_step
    bot.check_robot_state()


# Data reset
def data_reset():
    bot.NODE_EXISTED = get_scense_id(property_key)
    # bot.NODE_EXISTED = []
    bot.OBSERVED_NODE = []
    bot.RECORD_NODE = []
    bot.PATH_NODE = []
    bot.ALL_STEP = 0
    bot.SPECIAL_OBSERVED = [None]
    bot.SUCCESS_ARRIVE = False
    bot.SEARCH_MODE = False
    # bot.map_graph.delete_all()


def data_save():
    print("Data save in:", process_sub_epoch)
    real_path_dis = path_distance(bot.PATH_NODE)
    process_dic = {"map_exist": map_exist,
                   "SR": bot.SUCCESS_ARRIVE, "SPL": spl_value, "bot.ALL_STEP": bot.ALL_STEP,
                   "real_path_dis": real_path_dis, "shortest_path_dis": bot.shortest_path_value,
                   "star_position": star_position, "star_rotation": star_rotation,
                   "target_position": bot.ai2_final_xyz, "target_rotation": bot.ai2_final_rot,
                   "target_obj_id": bot.finally_target}
    process_data_save = pd.DataFrame(process_dic, index=[0])
    process_data_save.to_excel(process_sub_epoch + "/experiment_data.xlsx", index=False)
    with open(process_sub_epoch + "/experiment_data.txt", 'w') as f:
        data_write = "map_exist:" + str(map_exist) + "\n" + "SR:" + str(bot.SUCCESS_ARRIVE) + "\n" + \
                     "SPL:" + str(spl_value) + "\n" + "bot.ALL_STEP:" + str(bot.ALL_STEP) + "\n" + \
                     "real_path_dis:" + str(real_path_dis) + "\n" + "shortest_path_dis:" + \
                     str(bot.shortest_path_value) + "\n" + "star_position:" + str(star_position) + "\n" + \
                     "star_rotation:" + str(star_rotation) + "\n" + "target_position:" + \
                     str(bot.ai2_final_xyz) + "\n""target_rotation:" + str(bot.ai2_final_rot) + "\n" + \
                     "target_obj_id:" + str(bot.finally_target)
        f.write(data_write)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir += process_sub_epoch[1:]
        bot.export_neo4j_data(current_dir)
    print("data now is:\n", data_write)


# IKG build to record environment
def ikb_build():
    # print("bot.shortest_node_get:", bot.shortest_node_get)
    for short_i_, short_node_get_ in enumerate(bot.shortest_node_get):
        # print("short_node_get_:", short_node_get_)

        positions = bot.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        positions_get = []
        for pos_i_, pos_node_get in tqdm(enumerate(positions)):
            dis = np.linalg.norm(np.array([pos_node_get['x'], pos_node_get['z']]) - np.array(
                [short_node_get_['x'], short_node_get_['z']]))
            if dis < 0.1:
                positions_get.append(pos_node_get)
        # print("len of positions_get:", len(positions_get))

        agent_get = bot.controller.step(action="Stand")
        start_agent = agent_get.metadata['agent']
        now_pose = [start_agent['position']['x'], start_agent['position']['z']]
        for teleport_try in positions_get:
            teleport_event = bot.controller.step(
                action="Teleport",
                # position=dict(x=12.5, y=0.90, z=2.5),
                position=teleport_try,
                rotation=dict(x=0, y=90, z=0),
                # horizon=0,
                # standing=True,
            )
            tele_state = agent_get.metadata['agent']
            tele_pose = [tele_state['position']['x'], tele_state['position']['z']]
            dis_new_ = np.linalg.norm(np.array(now_pose) - np.array(tele_pose))
            if teleport_event.metadata['lastActionSuccess'] and dis_new_ > 0.01:
                # print("Transmission successful.")
                break
            # else:
            #     print("Transmission failed due to:", teleport_event.metadata['errorMessage'])

        # print('position', teleport_event.metadata['agent']['position'])
        for create_surround_ in range(4):
            # Get node from current view
            nodes_id = get_obj_visible()
            create_relationships(nodes_id)
            bot.event = bot.controller.step(action="RotateLeft")


def check_step(frame):
    if bot.ALL_STEP % max_save_step == 0:
        cv2.imwrite(process_sub_epoch + "/observe" + str(bot.ALL_STEP) + ".jpg", frame)
        draw_point(bot.ALL_STEP)
        graph_write(bot.ALL_STEP)


def graph_write(step):
    # init
    query = "MATCH (n)-[r]->(m) RETURN n,r,m"
    data = bot.map_graph.run(query)
    G = nx.DiGraph()
    G_shorttest = nx.DiGraph()
    top, bottom, left, right = [10] * 4
    boder_type = cv2.BORDER_CONSTANT
    boderColor = (0, 0, 0)
    node_colors = []
    node_colors_shoteest = []
    edge_colors = []
    edge_colors_shortest = []
    plt.figure(figsize=(6, 6))
    # select
    for record in data:
        # print(dict(record["n"])['Type'])
        G.add_node(dict(record["n"])['obj_id'])
        G.add_node(dict(record["m"])['obj_id'])
        if dict(record['r'])['euclidean_dis'] < 200:
            G.add_edge(dict(record["n"])['obj_id'], dict(record["m"])['obj_id'])
        if dict(record["n"])['obj_id'] in bot.navigation_path_obj_id and dict(record["m"])[
            'obj_id'] in bot.navigation_path_obj_id:
            edge_colors.append('red')
            edge_colors_shortest.append('black')
            G_shorttest.add_edge(dict(record["n"])['obj_id'], dict(record["m"])['obj_id'])
        else:
            edge_colors.append('black')
    pos = nx.random_layout(G)

    for node in G.nodes():
        if node != bot.finally_target and node not in bot.navigation_path_obj_id:
            node_colors.append('blue')
        elif node == bot.finally_target:
            node_colors.append('red')
            G_shorttest.add_node(node)
            node_colors_shoteest.append('red')
        elif node in bot.navigation_path_obj_id:
            G_shorttest.add_node(node)
            node_colors_shoteest.append('green')
            node_colors.append('green')
        else:
            node_colors.append('blue')
    pos_shorttest = nx.random_layout(G_shorttest)
    # graph
    nx.draw(G, pos=pos, with_labels=True, font_size=6, alpha=0.5, width=0.4, node_color=node_colors,
            edge_color=edge_colors)
    plt.savefig(process_sub_epoch + '/temp.png')
    G.clear()
    img = cv2.imread(process_sub_epoch + '/temp.png')
    img_copy = cv2.copyMakeBorder(img, top, bottom, left, right, boder_type, None, boderColor)
    cv2.imwrite(process_sub_epoch + "/graph" + str(step) + '.jpg', img_copy)
    plt.cla()
    # shorttest
    nx.draw(G_shorttest, pos=pos_shorttest, with_labels=True, font_size=6, alpha=0.5, width=0.4,
            node_color=node_colors_shoteest, edge_color=edge_colors_shortest)
    plt.savefig(process_sub_epoch + '/temp.png')
    G_shorttest.clear()
    img = cv2.imread(process_sub_epoch + '/temp.png')
    img_copy = cv2.copyMakeBorder(img, top, bottom, left, right, boder_type, None, boderColor)
    cv2.imwrite(process_sub_epoch + "/graph_shorttest" + str(step) + '.jpg', img_copy)
    plt.cla()


if __name__ == '__main__':
    # Experiment setting
    ai2thor_choice = False  # choose iThor or ProcTHOR
    plot_experiment_picture = False
    random_choose = False
    logger = logging.getLogger()
    logger.info('start')
    if ai2thor_choice:
        DELIMITER = '_'
    else:
        DELIMITER = '|'

    max_save_step = 25
    width_set = 600
    height_set = 600
    fov = 90
    file_path = 'a_star.yaml'
    with open(file_path, 'r', encoding='utf-8') as read:
        data = yaml.load(read, Loader=yaml.FullLoader)  # Config file
    version = data['version']
    amplify = data['amplify']
    radius = data['robot_radius']
    grid_size = data['grid_size']

    max_step = 500
    dataset = prior.load_dataset("procthor-10k")

    py_name = os.path.basename(__file__).split(".")[0][-1]
    epoch_index = int(py_name)
    epoch_bias = 0
    epoch_num = 50

    house = dataset['val'][0]
    process_name = "/process_data_eval"
    bot = Robotcontrol(house_env=house)
    database_choose = f'ikb-v{epoch_index}'
    bot.map_graph = Graph('neo4j://localhost:7687', auth=('neo4j', 'ab123456'), name=database_choose)
    bot.gds = GraphDataScience('neo4j://localhost:7687', auth=('neo4j', 'ab123456'), database=database_choose)
    bot.matcher = NodeMatcher(bot.map_graph)

    for house_ever in range(0, epoch_num):
        if plot_experiment_picture:
            house_num = 1
        else:
            house_num = int(epoch_bias + epoch_num * epoch_index) + house_ever
        if ai2thor_choice:
            house = "FloorPlan" + str(house_num)
        else:
            house = dataset['val'][house_num]
        # print("house:", house)

        bot.controller.reset(scene=house)
        positions = bot.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        bot.event = bot.controller.step(dict(action='LookDown', degrees=15))

        property_key = "obj_id"
        # nodes_id = []
        bot.NODE_EXISTED = get_scense_id(property_key)
        # Added a third view of the viewing camera
        third_img_get = get_top_down_frame()
        # cv2.imshow("third_img", third_img_get)
        # cv2.waitKey(1)
        bot.collect_now_node()
        print('build_map start')
        data_reset()  # data reset

        target_id_choose = get_target_id_assemble()
        for finally_target_get in target_id_choose:
            # Create the process path
            process_path = f"../DGN_eval_file/{process_name}"
            sub_path = "/epoch"
            EKB_state = "EKB_exist_" + str(bot.SEARCH_MODE is False)
            if not os.path.exists(process_path):
                os.makedirs(process_path)
            sub_num = len(os.listdir(process_path))
            # procthor
            process_eval_path = process_path + sub_path + str(sub_num + 1) + "_house_" + str(house_num) + EKB_state
            logger.info(f'epoch:{str(sub_num)},house:{str(house_num)}')

            # Reset the graph
            bot.map_graph.delete_all()

            # Insure the target exist
            bot.finally_target = finally_target_get
            positions = bot.controller.step(
                action="GetReachablePositions"
            ).metadata["actionReturn"]
            random.seed(10)
            position = random.choice(positions)

            if plot_experiment_picture:
                bot.map_graph.delete_all()
            for eva_i in range(2):
                bot.save_img_node_num = 0
                try:
                    teleport_event = bot.controller.step(
                        action="Teleport",
                        # position=dict(x=12.5, y=0.90, z=2.5),
                        position=position,
                        rotation=dict(x=0, y=90, z=0),
                        # horizon=0,
                        # standing=True,
                    )
                except ValueError as e:
                    warnings.warn("teleport error,stand init position:", e)
                    teleport_event = bot.controller.step(action="Stand")
                start_agent = teleport_event.metadata['agent']
                start_position = bot.real_2_cv_xyz(start_agent['position']['x'],
                                                   start_agent['position']['z'])
                start_frame = teleport_event.frame
                data_reset()  # data reset
                map_exist = "map_exist" if eva_i else "no_map"
                Evaluation_conditions = "pre_map"
                process_sub_epoch = process_eval_path + "/" + str(map_exist)
                # check_step(teleport_event.frame)
                if not os.path.exists(process_sub_epoch):
                    os.makedirs(process_sub_epoch)
                if plot_experiment_picture:
                    init_draw(start_position, start_frame)
                    graph_write(0)
                bot.check_robot_state()
                star_position = bot.agent['position']
                star_rotation = bot.agent['rotation']

                confirm_goal = check_target()  # Choose the target

                if not confirm_goal:
                    print("target not exist", bot.finally_target)
                    continue
                if map_exist == "map_exist":
                    logger.info('map_exist build map')
                    ikb_build()
                    map_find_shortest_flag = bot.find_shortest_path_nodes()
                    if map_find_shortest_flag:
                        map_exist = "map_exist"
                    else:
                        map_exist = "map_build_failed"
                    teleport_event = bot.controller.step(
                        action="Teleport",
                        # position=dict(x=12.5, y=0.90, z=2.5),
                        position=position,
                        rotation=dict(x=0, y=90, z=0),
                        # horizon=0,
                        # standing=True,
                    )
                try:
                    search_process()  # Search process
                except (TimeoutError, RecursionError):
                    # real_path_dis = path_distance(bot.PATH_NODE)
                    data_save()
                    continue
                bot.collect_now_node()
                if plot_experiment_picture:
                    event = bot.controller.step(action='Stand')
                    cv2.imwrite(process_sub_epoch + "/observe" + str(bot.ALL_STEP) + ".jpg", event.frame)
                    draw_point(bot.ALL_STEP)
                    graph_write(bot.ALL_STEP)
                draw_point()  # Draw the path
                spl_value = compute_single_spl(
                    bot.PATH_NODE,
                    bot.shortest_node_get,
                    bot.SUCCESS_ARRIVE
                )

                # real_path_dis = path_distance(bot.PATH_NODE)
                data_save()

                # Over the max step
                if bot.ALL_STEP > max_step:
                    print("Over max step")
                    break

                if not bot.SUCCESS_ARRIVE:
                    print("No map exist, and no target found")
                    continue
