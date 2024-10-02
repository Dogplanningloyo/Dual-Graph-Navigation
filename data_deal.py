
import csv
import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import MultipleLocator
import matplotlib.patches as patches
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import prior
from tqdm import tqdm
from collections import Counter

pd.set_option('display.max_columns', None)


# pd.set_option('display.max_rows',None)

# graph rebuild
def import_graph(epoch_name, map_choose):
    '''

    :param epoch_name: which epoch need import
    :param map_choose: no_map or map_exist
    :return:
    '''
    relative_path = "/process_data" + epoch_name + "/" + map_choose + "/graph_data.csv"

    # graph data deal
    node_name = "graph_node.csv"
    rele_name = "graph_rele.csv"
    relative_data_name = "graph_data.csv"
    map_graph = Graph('neo4j://localhost:7687', auth=('neo4j', 'ab123456'))
    absolut_path = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.join(absolut_path, )
    graph_data_file = os.path.join(current_dir, relative_data_name)
    graph_node_file = os.path.join(current_dir, node_name)
    graph_rele_file = os.path.join(current_dir, rele_name)

    graph_type = {
        "_id": str,
        "_start": str,
        "_end": str,
    }
    df_graph = pd.read_csv(graph_data_file, encoding="utf-8", sep=';', dtype=graph_type)
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

    # 
    cypher_import = r" CALL apoc.import.csv(" \
                    f"[{{fileName: '{graph_node_file}', labels: ['object']}}]," \
                    f"[{{fileName: '{graph_rele_file}', type: 'reachable'}}], " \
                    r"{delimiter: '|', arrayDelimiter: ',', stringIds: false, ignoreDuplicateNodes: true})"
    results_item = map_graph.run(cypher_import).data()


# 
def data_vis():
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False

    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'

    plt.figure(figsize=(20, 8), dpi=80)

    year = ['2015', '2016', '2017']
    data1 = [68826, 73125, 77198]
    data2 = [50391, 54455, 57892]

    # year, 
    x = range(len(year))

    plt.bar(x, data1, width=0.2, color='#FF6347')

    # 0.2, 0.2
    plt.bar([i + 0.2 for i in x], data2, width=0.2, color='#008B8B')

    # (, 0.1)
    plt.xticks([i + 0.1 for i in x], year)

    plt.ylabel('Score', size=13)

    color = ['#FF6347', '#008B8B']
    labels = ['data_1', 'data_2']

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # bbox_to_anchorlegend
    ax.legend(handles=patches, bbox_to_anchor=(0.65, 1.12), ncol=3)  # legend
    # 
    for x1, y1 in enumerate(data1):
        plt.text(x1, y1 + 200, y1, ha='center', fontsize=16)
    for x2, y2 in enumerate(data2):
        plt.text(x2 + 0.2, y2 + 200, y2, ha='center', fontsize=16)

    plt.show()


# 
def data_from_ai2thor():
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

    scenes = kitchens + living_rooms + bedrooms + bathrooms
    name_list = []

    for scene in tqdm(scenes, desc='Computing the house object'):
        controller = Controller(scene=scene,
                                platform=CloudRendering, )
        event = controller.step(action="Stand")
        objs_msg = event.metadata["objects"]
        controller.__exit__()
        for obj_msg in objs_msg:
            # str.split
            # obj_name = obj_msg["name"].split("_")[0]
            obj_name = obj_msg['objectType']
            # if obj_name not in name_list:
            name_list.append(obj_name)
            # print(f"{obj_name} append")
            obj_count = Counter(name_list)
            # print("The type of obj_count:",type(obj_count))
            # print("The count get is:",obj_count)
    data_deal_dirs = './data_statistics_deal/'

    if not os.path.exists(data_deal_dirs):
        os.makedirs(data_deal_dirs)


    with open(data_deal_dirs + "obj_name_statistics.csv", "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Obj_name', 'Frequency'])  # head
        writer.writerows(obj_count.items())  # content#


# pro_thor 
def data_from_prothor(data_deal_dirs='./data_statistics_deal/'):
    name_list = []
    dataset = prior.load_dataset("procthor-10k")

    for scene_num in tqdm(range(10000), desc='Computing the house object'):
        house = dataset['train'][scene_num]

        controller = Controller(scene=house,
                                platform=CloudRendering, )
        event = controller.step(action="Stand")
        objs_msg = event.metadata["objects"]
        controller.__exit__()
        for obj_msg in objs_msg:
            # str.split
            # obj_name = obj_msg["name"].split("_")[0]
            obj_name = obj_msg['objectType']
            # if obj_name not in name_list:
            name_list.append(obj_name)
            # print(f"{obj_name} append")
            obj_count = Counter(name_list)

    if not os.path.exists(data_deal_dirs):
        os.makedirs(data_deal_dirs)

    with open(data_deal_dirs + "pro_obj_name_statistics.csv", "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Obj_name', 'Frequency'])  # head
        writer.writerows(obj_count.items())  # content
    #     pickle.dump(name_list, fp)


# 
def compute_objects(dir_path, csv_file="obj_name_statistics.csv"):
    data_graph_path = dir_path + csv_file
    df_data = pd.read_csv(data_graph_path, encoding="utf-8")
    df_data = df_data.sort_values(by='Frequency', ascending=False)
    print("df_data:\n", df_data)

    df_data.plot.bar(x='Obj_name')
    plt.tight_layout()
    plt.savefig(dir_path + csv_file[:-4] + '_compute_ai2thor.jpg')
    plt.show()


# 
def statistics_data():
    process_dir = "2024_4_new_V1/process_data_procthor_DGN_V1"
    # process_dir = "2024_networkx_V1/process_data_procthor_3000"
    # process_dir = "2024_networkx_V1/process_save_data_3000"
    # process_dir = "2024_4_new_V1/process_data_procthor_KBN_5_test"
    # process_dir = "process_data_old_v1"
    file_all_get = os.listdir(process_dir)
    # print("file_all_get is:\n", file_all_get)
    print("file_all_len is:\n", len(file_all_get))
    all_eval_num = 0
    map_exist_all_eval_num = 0
    no_map_success_num = 0
    map_exist_success_num = 0
    no_map_spl_all = 0
    map_exist_spl_all = 0
    max_step = 0
    sum_step = 0
    max_step_exist = 0
    sum_step_exist = 0
    # random
    # map_exist_all_eval_num=1
    for every_file in file_all_get:
        no_map_path = os.path.join(process_dir, every_file, "no_map/experiment_data.xlsx")
        map_exist_path = os.path.join(process_dir, every_file, "map_exist/experiment_data.xlsx")
        # 
        # os.path.exists(map_exist_path)
        if os.path.exists(no_map_path) and os.path.exists(map_exist_path):
            all_eval_num += 1
            # all_eval_num += 1
            no_data_read = pd.read_excel(no_map_path)
            exist_data_read = pd.read_excel(map_exist_path)
            # if no_data_read['SR'][0]:
            if no_data_read['SR'][0]:
                no_map_success_num += 1
                no_map_spl_all += no_data_read['SPL'][0]
                sum_step += no_data_read['bot.ALL_STEP'][0]
                if int(no_data_read['bot.ALL_STEP'][0]) > max_step:
                    max_step = int(no_data_read['bot.ALL_STEP'])

        # if no_data_read['SR'][0]:


        # 
        # if os.path.exists(os.path.join(process_dir, every_file, "map_exist")):
            # map_exist_all_eval_num += 1
        # if os.path.exists(map_exist_path):
            # map_exist_all_eval_num += 1
            if exist_data_read["map_exist"][0] == "map_exist":
                map_exist_all_eval_num += 1
                if exist_data_read['SR'][0]:
                    map_exist_success_num += 1
                    map_exist_spl_all += exist_data_read['SPL'][0]
                    sum_step_exist += exist_data_read['bot.ALL_STEP'][0]
                    if int(exist_data_read['bot.ALL_STEP'][0]) > max_step_exist:
                        max_step_exist = int(exist_data_read['bot.ALL_STEP'])
            else:
                print("ikb，：",map_exist_path)
    no_map_SR = round(no_map_success_num / all_eval_num * 100, 2)
    map_exist_SR = round(map_exist_success_num / map_exist_all_eval_num * 100, 2)
    no_map_SPL = round(no_map_spl_all / all_eval_num * 100, 2)
    map_exist_SPL = round(map_exist_spl_all / map_exist_all_eval_num * 100, 2)
    map_improvements = round((map_exist_SPL - no_map_SPL) / no_map_SPL * 100, 2)
    print("No map SR is:", no_map_SR)
    print("Map exist SR is:", map_exist_SR)
    print(f"SPL:\nno map:{no_map_spl_all / all_eval_num}\nmap exist:{map_exist_spl_all / map_exist_all_eval_num}")
    print('max_step:', max_step)
    print('averge_step', sum_step / no_map_success_num)
    print('max_step_exist:', max_step_exist)
    print('averge_step_exist', sum_step_exist / map_exist_all_eval_num)
    plt.figure(figsize=(7, 10), dpi=80)

    compare_num = ['SR', 'SPL']
    no_map_data = [no_map_SR, no_map_SPL]
    map_exist_data = [map_exist_SR, map_exist_SPL]

    # , 
    bar_num = range(len(compare_num))
    columnar_width = 0.2

    plt.bar(bar_num, no_map_data, width=columnar_width, color='#FF6347')

    # 0.2, 0.2
    plt.bar([i + columnar_width for i in bar_num], map_exist_data, width=columnar_width, color='#008B8B')

    # (, 0.1)
    plt.xticks([i + columnar_width / 2 for i in bar_num], compare_num)

    plt.ylabel('Score', size=13)

    color = ['#FF6347', '#008B8B']
    labels = ['no_map', 'map_exist']
    y_major_locator = MultipleLocator(10)
    # y10，
    ax = plt.gca()
    # ax
    ax.yaxis.set_major_locator(y_major_locator)
    # x-0.511，0.5，，
    plt.ylim(-0.5, 100)

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]

    ax = plt.gca()
    ax.legend(handles=patches, bbox_to_anchor=(0.65, 1.12), ncol=3)  # legend
    # 
    for x1, y1 in enumerate(no_map_data):
        plt.text(x1, y1 + 3, y1, ha='center', fontsize=16)
    for x2, y2 in enumerate(map_exist_data):
        plt.text(x2 + columnar_width, y2 + 3, y2, ha='center', fontsize=16)

    data_deal_dirs = './data_statistics_deal/'
    plt.savefig(data_deal_dirs + 'statics_data_old_1.jpg')
    print("All test num is:", all_eval_num)
    print("Save path is:", data_deal_dirs + 'statics_data_new_1.jpg')

    plt.show()

# no map
def only_no_map_statistics_data():
    process_dir = "DGN_eval_file/process_data_eval"
    # process_dir = "2024_networkx_V1/process_data_procthor_3000"
    # process_dir = "2024_networkx_V1/process_save_data_3000"
    # process_dir = "2024_4_new_V1/process_data_procthor_KBN_5_test"
    # process_dir = "process_data_old_v1"
    file_all_get = os.listdir(process_dir)
    # print("file_all_get is:\n", file_all_get)
    print("file_all_len is:\n", len(file_all_get))
    all_eval_num = 0
    map_exist_all_eval_num = 0
    no_map_success_num = 0
    map_exist_success_num = 0
    no_map_spl_all = 0
    map_exist_spl_all = 0
    max_step = 0
    sum_step = 0
    max_step_exist = 0
    sum_step_exist = 0
    # random
    # map_exist_all_eval_num=1
    for every_file in file_all_get:
        no_map_path = os.path.join(process_dir, every_file, "no_map/experiment_data.xlsx")
        map_exist_path = os.path.join(process_dir, every_file, "map_exist/experiment_data.xlsx")
        # 
        # os.path.exists(map_exist_path)
        if os.path.exists(no_map_path):
            all_eval_num += 1
            # all_eval_num += 1
            no_data_read = pd.read_excel(no_map_path)
            # exist_data_read = pd.read_excel(map_exist_path)
            # if no_data_read['SR'][0]:
            if no_data_read['SR'][0]:
                no_map_success_num += 1
                no_map_spl_all += no_data_read['SPL'][0]
                sum_step += no_data_read['bot.ALL_STEP'][0]
                if int(no_data_read['bot.ALL_STEP'][0]) > max_step:
                    max_step = int(no_data_read['bot.ALL_STEP'])
                    print("max step eposid is:",no_map_path)

        # if no_data_read['SR'][0]:


        # 
        # if os.path.exists(os.path.join(process_dir, every_file, "map_exist")):
            # map_exist_all_eval_num += 1
        if os.path.exists(map_exist_path):
            # map_exist_all_eval_num += 1
            exist_data_read = pd.read_excel(map_exist_path)

            if exist_data_read["map_exist"][0] == "map_exist":
                map_exist_all_eval_num += 1
                if exist_data_read['SR'][0]:
                    map_exist_success_num += 1
                    map_exist_spl_all += exist_data_read['SPL'][0]
                    sum_step_exist += exist_data_read['bot.ALL_STEP'][0]
                    if int(exist_data_read['bot.ALL_STEP'][0]) > max_step_exist:
                        max_step_exist = int(exist_data_read['bot.ALL_STEP'])
            else:
                print("ikb，：",map_exist_path)
    no_map_SR = round(no_map_success_num / all_eval_num * 100, 2)
    map_exist_SR = round(map_exist_success_num / map_exist_all_eval_num * 100, 2)
    no_map_SPL = round(no_map_spl_all / all_eval_num * 100, 2)
    map_exist_SPL = round(map_exist_spl_all / map_exist_all_eval_num * 100, 2)
    map_improvements = round((map_exist_SPL - no_map_SPL) / no_map_SPL * 100, 2)
    print("No map SR is:", no_map_SR)
    print("Map exist SR is:", map_exist_SR)
    print(f"SPL:\nno map:{no_map_spl_all / all_eval_num}\nmap exist:{map_exist_spl_all / map_exist_all_eval_num}")
    print('max_step:', max_step)
    print('averge_step', sum_step / no_map_success_num)
    print('max_step_exist:', max_step_exist)
    print('averge_step_exist', sum_step_exist / map_exist_all_eval_num)
    plt.figure(figsize=(7, 10), dpi=80)

    compare_num = ['SR', 'SPL']
    no_map_data = [no_map_SR, no_map_SPL]
    map_exist_data = [map_exist_SR, map_exist_SPL]

    # , 
    bar_num = range(len(compare_num))
    columnar_width = 0.2

    plt.bar(bar_num, no_map_data, width=columnar_width, color='#FF6347')

    # 0.2, 0.2
    plt.bar([i + columnar_width for i in bar_num], map_exist_data, width=columnar_width, color='#008B8B')

    # (, 0.1)
    plt.xticks([i + columnar_width / 2 for i in bar_num], compare_num)

    plt.ylabel('Score', size=13)

    color = ['#FF6347', '#008B8B']
    labels = ['no_map', 'map_exist']
    y_major_locator = MultipleLocator(10)
    # y10，
    ax = plt.gca()
    # ax
    ax.yaxis.set_major_locator(y_major_locator)
    # x-0.511，0.5，，
    plt.ylim(-0.5, 100)

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]

    ax = plt.gca()
    ax.legend(handles=patches, bbox_to_anchor=(0.65, 1.12), ncol=3)  # legend
    # 
    for x1, y1 in enumerate(no_map_data):
        plt.text(x1, y1 + 3, y1, ha='center', fontsize=16)
    for x2, y2 in enumerate(map_exist_data):
        plt.text(x2 + columnar_width, y2 + 3, y2, ha='center', fontsize=16)

    data_deal_dirs = './data_statistics_deal/'
    if not os.path.exists(data_deal_dirs):
        os.makedirs(data_deal_dirs)
    plt.savefig(data_deal_dirs + 'only_no_map_statics_data_old_1.jpg')
    print("All test num is:", all_eval_num)
    print("Save path is:", data_deal_dirs + 'only_no_map_statics_data_new_1.jpg')
    plt.show()

if __name__ == '__main__':
    # absolute_paht = "/home/ab1231456/.config/Neo4j Desktop/Application/relate-data/dbmss/dbms-f8f450f8-124b-4bb1-891f-f5c6f40437b0/import/"
    print("Reading data deal")
    # statistics_data()
    only_no_map_statistics_data()

