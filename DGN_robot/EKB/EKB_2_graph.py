# -*- coding: utf-8 -*-
# @Time    :2024/2/27 22:07
# @Author  :LSY Dreampoet
# @SoftWare:PyCharm
import os
import pickle

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher
from tqdm import tqdm

global pos


# networkx构建图谱
def create_graph_with_networkx(G):
    global pos  # 保证生成位置的一致性

    # 将ekb_data中的数据添加到图中
    for i in range(ekb_data.shape[0]):
        # print("ekb_data['name'][i]:",ekb_data['name'][i])
        # G.add_node(i, name=ekb_name[i])
        for j in range(i, ekb_data.shape[1]):
            if ekb_data[i][j] >= find_rate:
                G.add_node(i, name=ekb_name[i])
                G.add_node(j, name=ekb_name[j])
                G.add_edge(i, j, weight=ekb_data[i][j])
    # 画图
    # 创建对应的文件夹
    save_graph_dir = "../data_save/graph"
    if os.path.exists(save_graph_dir) == False:
        os.makedirs(save_graph_dir)
    img_len = len(os.listdir(save_graph_dir))
    ekb_save_path = os.path.join(save_graph_dir, f"EKB_graph_{img_len}.png")
    # labels = nx.get_node_attributes(G, 'name')
    # 设置图的大小
    plt.figure(figsize=(10, 13))
    # target = 50
    # node_colors = ["red" if n == target else (0.3, 0.5, 1) for n in G.nodes()]

    # 使用spring_layout布局算法
    # pos = nx.spring_layout(G)
    pos = nx.random_layout(G)
    # pos = nx.multipartite_layout(G, subset_key="layer",)
    # pos = nx.kamada_kawai_layout(G)
    # 获取目标节点的所有相邻节点
    neighbors = list(nx.neighbors(G, target))

    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1500, node_color=(0.3, 0.5, 1))
    # 创建一个子图，只包含目标节点和它的所有相邻节点
    H = G.subgraph([target] + neighbors)

    # 绘制子图中的节点
    nx.draw_networkx_nodes(H, pos, node_color=(0.3, 0.5, 0.9), node_size=1500, edgecolors=(0.5, 0.9, 0.5), linewidths=5)

    # 绘制子图中的边
    edge_colors = ['red' if e[0] == target or e[1] == target else (0, 0, 0) for e in H.edges()]
    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, width=5)
    nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color=(1, 0.5, 0.1), node_size=1500,
                           edgecolors=(0.5, 0.9, 0.5),
                           linewidths=5)  # 绘制节点1，颜色为绿色，轮廓颜色为红色，轮廓宽度为2

    # nx.draw_networkx_labels(G, pos=nx.spring_layout(G), labels=labels)
    plt.savefig(ekb_save_path, transparent=True)
    plt.show()
    plt.close()


# 使用neo4j构建图谱
def create_graph_with_neo4j(graph):
    # 创建节点
    node_all_get = []
    print("ekb_data:", ekb_data.shape[0])
    for i_n in range(ekb_data.shape[0]):
        # graph.run("CREATE (a:Node {name: {name}})", name=ekb_name[i])
        node = Node("label", name=ekb_name[i_n])
        graph.create(node)
        node_all_get.append(node)
    print("node_all_get:", len(node_all_get))
    # 创建关系 TODO 需要优化
    # 使用tqdm显式角度
    for i_r in tqdm(range(ekb_data.shape[0])):
        # for i_r in tqdm(range(ekb_data.shape[0])):
        for j_r in range(i_r, ekb_data.shape[0]):
            if ekb_data[i_r][j_r] >= 0.3:
                # 　neo4j 创建无向边

                # print("ekb_data[i_r][j_r]:", ekb_data[i_r][j_r])
                rel = Relationship(node_all_get[i_r], "CONNECTED", node_all_get[j_r], weight=float(ekb_data[i_r][j_r]))
                graph.create(rel)
    # pass


# 查找节点关系
def node_neighbor(target_obj):
    global pos  # 保证生成位置的一致性
    # 　查找节点关系
    neighbor_dic = {}  # 获得相邻节点的字典（可覆盖）
    # target_obj = 1
    neighbors = list(nx.neighbors(ekb_G, target_obj))
    connect_nodes = []
    for neighbor in neighbors:

        # 直接打印邻居节点的name属性
        edge_cost = ekb_G[target_obj][neighbor]['weight']
        node_id = ekb_G.nodes[neighbor]['name']
        # print(f"Neighbor of 1: {node_id}, Relationship: {edge_cost}")
        neighbor_dic[edge_cost] = node_id
        if edge_cost >= find_rate:  # 设置对应的关系阈值
            connect_nodes.append(neighbor)
        # 保存对应的节点和边权重，为键值对
    print("neighbor_dic is:", neighbor_dic)
    sorted_dict = {k: v for k, v in sorted(neighbor_dic.items(), key=lambda item: item[0], reverse=True)}
    # color_get = ['green','blue']
    print("sorted_dict:", sorted_dict)
    # 将target_obj节点与相邻节点都绘制出来
    # 创建一个子图，包含这个节点和它的所有相邻节点
    # connect_nodes = connect_nodes[::7]
    H = ekb_G.subgraph([target_obj] + connect_nodes)

    # H = ekb_G.edge_subgraph([(target_obj, n) for n in connect_nodes])
    edge_colors = ['red' if e[0] == target or e[1] == target else (0, 0, 0) for e in H.edges()]

    # node_colors = ["red" if n == target else (0.3, 0.5, 1) for n in H.nodes()]
    edge_widths = [((d['weight']-find_rate)*5)  for (u, v, d) in H.edges(data=True)]
    print("edge_widths:",edge_widths)
    # print("node_colors:",node_colors)
    # 绘制子图
    # pos = nx.spring_layout(ekb_G)

    plt.figure(figsize=(10, 10))
    nodes_size = 5000
    # 绘制子图中的节点
    nx.draw_networkx_nodes(H, pos, node_color=(0.3, 0.5, 0.9), node_size=nodes_size, edgecolors=(0.5, 0.9, 0.5))
    # 创建对应的文件夹
    save_graph_dir = "../data_save/graph"
    if os.path.exists(save_graph_dir) == False:
        os.makedirs(save_graph_dir)
    img_len = len(os.listdir(save_graph_dir))
    save_path = os.path.join(save_graph_dir, f"graph_{img_len}.png")

    nx.draw(H, pos, with_labels=False, node_color=(0.3, 0.5, 0.9), node_size=nodes_size, edgecolors=(0.5, 0.9, 0.5), width=edge_widths,linewidths=10)
    nx.draw_networkx_edges(H, pos, edge_color=edge_colors, width=10)
    nx.draw_networkx_labels(H, pos, font_size=30)
    nx.draw_networkx_nodes(H, pos, nodelist=[target], node_color=(1, 0.5, 0.1), node_size=nodes_size,
                           edgecolors=(0.5, 0.9, 0.5),
                           linewidths=10) # 绘制特定节点
    # plt.savefig("../data_save/Target_only_edges_graph.png", transparent=True)
    plt.savefig(save_path, transparent=True)
    plt.show()
    plt.close()

from ultralytics import YOLO

if __name__ == '__main__':
    print('EKB_2_graph.py start ')
    # 读取pkl数据
    # ekb_data = pd.read_pickle('near_rate_array.pkl')
    # ekb_name = pd.read_pickle('all_obj_name.pkl')
    ekb_data = np.load("../ekb_rate_get.npy") # 读取EKB概率数据
    # name_mapping_dic = np.load('../EKB/name_mapping.npy', allow_pickle=True).item()
    # 加载yolo模型与特征提取器，比较器
    yolo_det = YOLO('yolov8n.pt')
    # 输出模型的识别id与类别
    obj_model_info = yolo_det.names
    print("obj_model_info_all:", obj_model_info)
    ekb_name = list(obj_model_info.values()) # 读取EKB名称

    print("ekb_name is:", ekb_name)
    print("chair in ekb is:", ekb_name.index("chair"))

    obj_num = len(ekb_name)
    find_rate = 1 / (obj_num * 7)  # 设置概论下限为平均类别，再下降20%的余量
    target = 56 # 特定目标

    # 　只取4行4列
    # ekb_data = ekb_data[:4,:4]
    # ekb_name = ekb_name[:4]
    # 生成图
    ekb_G = nx.Graph()
    # ekb_data = ekb_data[:10,:10]
    ekb_data = ekb_data
    # 关联neo4j数据库
    create_graph_with_networkx(ekb_G)
    # 保存EKB图谱为pkl文件
    with open("near_rate_graph.pkl", "wb") as f:
        pickle.dump(ekb_G, f)
    # 读取EKB图谱文件
    with open("near_rate_graph.pkl", "rb") as file:
        ekb_get = pickle.load(file)
    print("ekb_G is:", ekb_G)
    print("ekb_get is:", ekb_get)
    ekb_G = ekb_get

    # node_neighbor(50)
    node_neighbor(target)

    # 获取与该节点相连的所有关系

    # print("data is:", ekb_data)
