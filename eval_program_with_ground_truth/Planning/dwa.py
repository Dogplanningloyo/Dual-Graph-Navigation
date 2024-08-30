import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
# from plan import *
import yaml

radius = 20.0

path = 'a_star.yaml'
with open(path, 'r', encoding='utf-8') as read:
    data = yaml.load(read, Loader=yaml.FullLoader)  # Configuration file
# p = np.load('local_map.npy')
amplify = data['amplify']


class config:
    def __init__(self,
                 v_max=40.0,
                 v_min=-40.0,
                 w_min=-90 * pi / 180,
                 w_max=90 * pi / 180,
                 acc_linear=40.0,
                 w_acc=90 * pi / 180,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 v_reso=10.0,
                 w_reso=10 * pi / 180.0,
                 radius=40.0,
                 dt=0.2,
                 t=4.0
                 ):
        self.v_max = v_max
        self.v_min = v_min
        self.w_min = w_min
        self.w_max = w_max
        self.acc_linear = acc_linear
        self.w_acc = w_acc
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.v_reso = v_reso
        self.w_reso = w_reso
        self.radius = radius
        self.t = t
        self.dt = dt


class dwa:
    def __init__(self, config):
        self.config = config

    def distance_cost(self, goal, end):
        return sqrt((end[1] - goal[1]) ** 2 + (end[0] - goal[0]) ** 2)

    def speed_cost(self, v_final):
        return self.config.v_max - v_final

    def obstacle_cost(self, traj, map):
        MIN_COST = float('Inf')
        obstacle_x, obstacle_y = map[:, 0], map[:, 1]
        # obstacle_x, obstacle_y = np.where(map > 0)
        # print(obstacle_x, obstacle_y)
        # print(map)
        t1 = time.time()
        for i in range(len(traj)):
            cost = sqrt(np.min((obstacle_x - traj[i][0]) ** 2 + (obstacle_y - traj[i][1]) ** 2))
            if cost < MIN_COST:
                MIN_COST = cost
            if MIN_COST <= self.config.radius:
                return float('Inf')
        return 1 / MIN_COST

    def get_interval(self, v, w):
        v_max = v + self.config.acc_linear * self.config.dt
        v_min = v - self.config.acc_linear * self.config.dt
        w_min = w - self.config.w_acc * self.config.dt
        w_max = w + self.config.w_acc * self.config.dt
        return np.array([max(v_min, self.config.v_min), min(v_max, self.config.v_max)]), \
               np.array([max(w_min, self.config.w_min), min(w_max, self.config.w_max)])

    def motion(self, state, before):  # state
        state[0] += before[0] * self.config.dt * cos(state[2])
        state[1] += before[0] * self.config.dt * sin(state[2])
        state[2] += before[1] * self.config.dt
        state[3] = before[0]
        state[4] = before[1]
        return state

    def traj_get(self, state, before):
        times = 0
        traj = np.array(state)
        new_state = np.array(state)
        while times <= self.config.t:
            new_state = self.motion(new_state, before)
            traj = np.vstack((traj, new_state))
            times += self.config.dt
        return traj, new_state

    def planning(self, state, before, goal, map):
        v_interval, w_interval = self.get_interval(state[3], state[4])
        # print(v_interval, w_interval)
        best_traj = np.array(state)
        best_score = 10000
        for v in np.arange(v_interval[0], v_interval[1], self.config.v_reso):  # 
            for w in np.arange(w_interval[0], w_interval[1], self.config.w_reso):  # 
                traj, final = self.traj_get(state, [v, w])
                goal_score = self.distance_cost(goal, final)
                vel_score = self.speed_cost(final[3])
                obs_score = self.obstacle_cost(traj, map)
                # print('score:',goal_score,vel_score,obs_score)
                score = self.config.alpha * goal_score + self.config.beta * vel_score + self.config.gamma * obs_score
                if score <= best_score:
                    best_score = score
                    before = np.array([v, w])
                    best_traj = traj
        # print(best_traj)
        return best_traj, before


def path_optimize(map, start=np.array([0, 0, 45 * pi / 180, 0, 0]), u=np.array([0, 0]), goal=np.array([8, 8])):
    global_tarj = np.array(start)
    Dwa = dwa(config(radius=radius))

    while sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2) > radius:  # 1000，while
        t1 = time.time()
        current, u = Dwa.planning(start, u, goal, map)
        start = Dwa.motion(start, u)
        # print('x:',x)
        global_tarj = np.vstack((global_tarj, start))  # 
        print('dis:', sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2))
        t2 = time.time()
        print(t2 - t1)
        # print('x:', np.array(start))

    return global_tarj


if __name__ == '__main__':
    print("start dwa")
    # ox, oy, map, maps = handing_data(p)
    # x, y = ox.reshape(-1, 1), oy.reshape(-1, 1)
    # a_star = AStarPlanner(ox, oy, grid_size, map_get=maps)
    # path = start_with_oxoy([0, 0], [128, 325], ox, oy, map, maps, a_star)
    # maps = np.hstack((x, y))
    # global_tarj = np.array([-2.5, 47, 45 * pi / 180, 0, 0])
    # for i in range(len(path) - 1):
    #     global_tarj = np.vstack(
    #         (global_tarj, path_optimize(map=maps, start=np.array([path[i, 1], path[i, 0], 90 * pi / 180, 0, 0]),
    #                                     goal=np.array([path[i + 1, 1], path[i + 1, 0]]))))
    #
    # plt.plot(maps[:, 0], maps[:, 1], '.k')
    # plt.plot(global_tarj[:, 0], global_tarj[:, 1], '-r')
    # plt.show()
