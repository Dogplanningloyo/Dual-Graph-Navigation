"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

yaml_path = 'a_star.yaml'
with open(yaml_path, 'r', encoding='utf-8') as read:
    data = yaml.load(read, Loader=yaml.FullLoader)  # 配置文件
grid_size = data['grid_size']  # [m]
robot_radius = data['robot_radius']
show_animation = data['show_animation']

__version__ = 1.0


class AStarPlanner:
    def __init__(self, map_info, resolution, rr=robot_radius, w=0, h=0, map_get=None):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]?
        oy: y position list of Obstacles [m]?
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.w = w
        self.h = h
        self.resolution = resolution
        self.rr = rr * 2
        # fixme why min_y = 0?
        self.min_x, self.max_x, self.min_y, self.max_y = map_info
        self.min_y = 0
        # self.min_x, self.min_y = 0, 0
        # self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(map_get)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        # t_set_1 = time.time()
        while 1:
            if len(open_set) == 0:
                # print("Open set is empty..")
                return [0, 0], [0, 0], False

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # 判定条件适当放宽，可以有效减少到目标点的探索，还能避免报错
            judge_num = int(robot_radius // grid_size + 2)  # 通过网格和车位大小两者联系，来作为是否到位置的判断，这部分之前乘以了个2作为直径
            # print("judge_Num:",judge_num)
            if abs(current.x - goal_node.x) + abs(current.y - goal_node.y) < judge_num:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                # print("启发搜索时间：", time.time() - t_set_1)
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry, True

    # FIXME 可优化的地方
    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        # d = abs(n1.x - n2.x) + abs(n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        # collision check 用于判断是否超界
        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:  # FIXME 需要确定下具体x和y
            return False

        return True

    # TODO 探索函数，可以考虑特殊切割优化来大幅减少点数
    def calc_obstacle_map(self, map_get):  # 如果是为了构造地图，为啥不直接用数据直接组成？
        if map_get is None:
            return None
        # 设置网格大小和阈值
        grid_size = int(self.resolution)
        threshold = 55
        # if self.w != 0:
        #     self.min_x = 0
        #     self.min_y = 0
        #     self.max_x = round(self.w)
        #     self.max_y = round(self.h)
        # else:
        # FIXME You want to get the map shape, why not use map_get.shape directly?
        # self.min_x = round(min(ox))
        # self.min_y = 0  # 你也不能看到后面对吧
        # self.max_x = round(max(ox))
        # self.max_y = round(max(oy))
        #     # self.min_x1 = round(min(ox/grid_size))
        #     # self.min_y1 = 0  # 你也不能看到后面对吧
        #     # self.max_x1 = round(max(ox/grid_size))
        #     # self.max_y1 = round(max(oy/grid_size))
        #
        # self.x_width = round((self.max_x - self.min_x) / self.resolution)
        # self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # 获取图像尺寸和网格数量
        h, w = map_get.shape
        self.y_width = round(h / self.resolution)
        self.x_width = round(w / self.resolution)
        nrows = h // grid_size + 1
        ncols = w // grid_size + 1

        # 将图像划分为相应大小的网格
        grid = np.zeros((ncols, nrows), dtype=np.uint8)
        map_view = np.zeros((ncols, nrows), dtype=np.uint8)
        for i in range(nrows):
            for j in range(ncols):
                row_start = i * grid_size
                col_start = j * grid_size
                row_end = row_start + grid_size
                col_end = col_start + grid_size
                roi = map_get[row_start:row_end, col_start:col_end]
                pixel_count = cv2.countNonZero(roi)
                map_view[j, i] = pixel_count
                if pixel_count > threshold:
                    grid[j, i] = 255
        # if True:
        #     plt.title("grid_map")
        #     indexes = np.nonzero(grid)
        #     # print("indexes:",indexes)
        #     plt.scatter(indexes[0],indexes[1])
        #     plt.show()
        self.obstacle_map = grid

    @staticmethod
    def get_motion_model():
        # dx, dy, cost # 鼓励先向后前再向左右
        motion = [[1, 0, 2],
                  [0, 1, 1],
                  [-1, 0, 2],
                  [0, -1, 1], ]
        #   [-1, -1, math.sqrt(2)],
        #   [-1, 1, math.sqrt(2)],
        #   [1, -1, math.sqrt(2)],
        #   [1, 1, math.sqrt(2)]]

        return motion

    # A* 导航迭代器
    def a_star_plan(self, start, goal, show=show_animation):
        # print(__file__ + " start!!")
        sx = start[0]  # [m]
        sy = start[1]  # [m]
        gx = goal[0]  # [m]
        gy = goal[1]  # [m]

        rx, ry, judge = self.planning(sx, sy, gx, gy)  # 反馈父节点，作为路径
        # TODO you need to now which one should be draw
        # if judge:
        if show:  # pragma: no cover
            indexes = np.nonzero(self.obstacle_map)  # Find the obstacles index
            indexes = np.array(indexes)
            indexes *= self.resolution
            indexes[0] += self.min_x
            indexes[1] += self.min_y
            # indexes = self.obstacle_map
            plt.scatter(indexes[0], indexes[1])
            plt.scatter(goal[0], goal[1])
            # plt.plot((np.array(rx) - self.min_x), (np.array(ry) - self.min_y) ,
            #          '-d')
            plt.plot(rx, ry, marker='p', color='coral')
            # 去除坐标轴与背景
            plt.axis('off')
            plt.savefig('a_star_get.png', transparent=True, dpi=900)
            plt.show()
        path = np.stack((rx, ry), axis=1)
        path = np.flip(path, axis=0)

        return path, judge
