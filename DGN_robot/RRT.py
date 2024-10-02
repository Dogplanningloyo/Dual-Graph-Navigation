import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# 定义节点类
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


# 定义RRT类
class RRT_Planner:
    def __init__(self, start, goal, obstacles, map_info, max_iter=500, step_size=5, goal_sample_rate=0.5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        # self.x_lim = x_lim
        # self.y_lim = y_lim
        self.min_x, self.max_x, self.min_y, self.max_y = map_info
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.tree = [self.start]
        # self.obstacle_map_get(obstacles)

    def obstacle_map_get(self,map_get):
        threshold = 55
        # 获取图像尺寸和网格数量
        h, w = map_get.shape
        # y_width = round(h / self.resolution)
        # x_width = round(w / self.resolution)
        nrows = h // self.step_size + 1
        ncols = w // self.step_size + 1
        # 将图像划分为相应大小的网格
        grid = np.zeros((ncols, nrows), dtype=np.uint8)
        map_view = np.zeros((ncols, nrows), dtype=np.uint8)
        for i in range(nrows):
            for j in range(ncols):
                row_start = i * self.step_size
                col_start = j * self.step_size
                row_end = row_start + self.step_size
                col_end = col_start + self.step_size
                roi = map_get[row_start:row_end, col_start:col_end]
                pixel_count = cv2.countNonZero(roi)
                map_view[j, i] = pixel_count
                if pixel_count > threshold:
                    grid[j, i] = 255
        self.obstacles = grid.T
        self.min_x = int(self.min_x / self.step_size)
        self.max_x = int(self.max_x / self.step_size)
        self.min_y = int(self.min_y / self.step_size)
        self.max_y = int(self.max_y / self.step_size)
        plt.imshow(self.obstacles)
        plt.show()

    def is_collision(self, node):
        # for (ox, oy, size) in self.obstacles:
        #     if math.hypot(node.x - ox, node.y - oy) <= size:
        if self.obstacles[node.y][node.x-self.min_x] == 255:
            return True
        return False

    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            return Node(random.uniform(self.min_x,self.max_x), random.uniform(self.min_y, self.max_y))
        else:
            return Node(self.goal.x, self.goal.y)

    def get_nearest_node(self, random_node):
        return min(self.tree, key=lambda node: math.hypot(node.x - random_node.x, node.y - random_node.y))

    def plan(self,show=False):
        for _ in range(self.max_iter):
            random_node = self.get_random_node()
            nearest_node = self.get_nearest_node(random_node)
            theta = math.atan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)

            # new_node = Node(int((nearest_node.x + self.step_size * math.cos(theta))/self.step_size),
            #                 int((nearest_node.y + self.step_size * math.sin(theta))/self.step_size))
            new_node = Node(int((nearest_node.x + self.step_size * math.cos(theta))),
                            int((nearest_node.y + self.step_size * math.sin(theta))))
            new_node.parent = nearest_node

            if not self.is_collision(new_node):
                self.tree.append(new_node)

                # if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.step_size:
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.step_size:
                    self.goal.parent = new_node
                    self.tree.append(self.goal)
                    if show:
                        self.draw_graph(self.generate_final_course())
                    return self.generate_final_course(),True
        return [[0,0],[0,0]], None

    def generate_final_course(self):
        path = []
        node = self.goal
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]

    def draw_graph(self, path=None):
        plt.figure()
        for node in self.tree:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
            plt.plot(node.x, node.y, "bo")
            plt.text(node.x, node.y, f'({int(node.x)},{int(node.y)})', fontsize=8, ha='right')

        # for (ox, oy, size) in self.obstacles:
        #     circle = plt.Circle((ox, oy), size, color="r")
        #     plt.gca().add_patch(circle)
        np_indexes = np.nonzero(self.obstacles)  # Find the obstacles index
        # indexes *= self.step_size
        # plt.imshow(self.obstacles, cmap='gray')
        # indexes *= self.step_size
        indexes_x =np_indexes[1] + self.min_x
        indexes_y =np_indexes[0] + self.min_y
        # indexes = self.obstacle_map
        plt.scatter(indexes_x, indexes_y)
        # plt.scatter(goal[0], goal[1])

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r", linewidth=2)

        plt.plot(self.start.x, self.start.y, "bo")
        plt.plot(self.goal.x, self.goal.y, "bo")
        plt.text(self.start.x, self.start.y, f'({int(self.start.x)},{int(self.start.y)})', fontsize=8, ha='right')
        plt.text(self.goal.x, self.goal.y, f'({int(self.goal.x)},{int(self.goal.y)})', fontsize=8, ha='right')
        plt.xlim(self.min_x)
        plt.ylim(self.min_y)
        plt.grid(True)
        plt.show()


if __name__ == '__main__':

    # 参数设置
    # [-282, 177, 0, 372]
    # 读取maps_path_2d,maps_info
    maps_path_2d = np.load('maps_path_2d.npy')
    map_info = np.load('maps_info.npy')



    start = [0, 0]
    goal = [-90, 100]
    # obstacles = [(20, 20, 10), (40, 40, 10), (60, 60, 10)]
    obstacles = maps_path_2d
    # x_lim = [maps_info[0], maps_info[1]]
    # y_lim = [maps_info[2], maps_info[3]]
    max_iter = 500
    step_size = 10
    goal_sample_rate = 0.5

    rrt_1 = time.time()
    rrt = RRT_Planner(start, goal, obstacles, map_info, max_iter, step_size, goal_sample_rate)
    path,judge = rrt.plan()
    rrt_2 = time.time()
    print("Cost time:", rrt_2 - rrt_1)
    rrt.draw_graph(path)
