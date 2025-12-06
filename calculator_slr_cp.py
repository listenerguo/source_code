import copy
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
import json


def compute_rank(graph, communication_time, computation_time):
    task_rank = []
    task_number = len(graph)

    for task in range(task_number):
        if not graph[task]["predecessors"]:
            graph[task]["downward_rank"] = 0
        else:
            max_download = temp = 0
            for predecessor in graph[task]["predecessors"]:
                temp = graph[predecessor]["downward_rank"] + computation_time[predecessor] + communication_time[(predecessor, task)]
                if temp > max_download:
                    max_download = temp
            graph[task]["downward_rank"] = round(max_download*10)/10
    # upward_rank
    for task in reversed(range(task_number)):
        if not graph[task]["successors"]:
            graph[task]["upward_rank"] = computation_time[task]
        else:
            max_upward = temp = 0
            for successor in graph[task]["successors"]:
                temp = communication_time[(task, successor)] + graph[successor]["upward_rank"]
                if temp > max_upward:
                    max_upward = temp

            up_value = computation_time[task] + max_upward
            graph[task]["upward_rank"] = round(up_value*10)/10
        graph[task]["rank"] = round(graph[task]["upward_rank"] + graph[task]["downward_rank"], 1)
    for t, info in graph.items():
        task_rank.append((t, info["rank"]))
    max_value = max(task_rank, key=lambda x: x[1])[1]

    task_rank = sorted(task_rank, key=lambda x: x[1], reverse=True)
    scheduling_list = [x[0] for x in task_rank]

    return scheduling_list, max_value


def claculation_algo(tasks, edges, exe_speeds, com_speeds):
    mean_w = []
    for task in tasks:
        sum_exe = 0
        for speed in exe_speeds:
            sum_exe += round((task["data"]/speed)*10)/10
        mean_w.append(sum_exe/len(exe_speeds))
    mean_c = {}
    sum_com_speed = 0
    for speed in com_speeds:
        sum_com_speed += sum(speed)
    for edge, data in edges.items():
        mean_c[edge] = round((data/sum_com_speed)*10)/10

    graph = {}
    for task in range(len(tasks)):
        graph[task] = {"id": tasks[task]['id'], "layer_level": tasks[task]['layer_rank'], "upward_rank": 0,
                       "downward_rank": 0, 'rank': 0,
                       "successors": [], "predecessors": [], "start_time": None, "end_time": 0, "mapping": None}

    for edge, _ in edges.items():
        predecessor = edge[0]
        successor = edge[1]
        graph[predecessor]["successors"].append(successor)
        graph[successor]["predecessors"].append(predecessor)

    scheduling_list, cp_max = compute_rank(graph, mean_c, mean_w)

    segments = []
    prev_layer_level = None
    for id, value in graph.items():
        layer_level = value["layer_level"]
        if layer_level != prev_layer_level:
            segments.append([id])
        else:
            segments[-1].append(id)
        prev_layer_level = layer_level

    cp_lists = [0]
    for same_level in segments:
        for i in same_level:
            if graph[i]["rank"] - cp_max<= 0.1 and  cp_lists[-1] in  graph[i]["predecessors"]:
                cp_lists.append(i)
                break

    sum_cp_exe = []
    for process in range(len(exe_speeds)):
        sum_cp_exe.append(0)
        for task in cp_lists:
            sum_cp_exe[-1] += round(tasks[task]['data'] / exe_speeds[process], 1)

    min_process_id = [i for i, x in enumerate(sum_cp_exe) if x == min(sum_cp_exe)]
    process_cp = random.choice(min_process_id)

    speed_up_fenzi = 0
    max_speed = max(exe_speeds)
    for i in tasks:
        exe = round(i["data"]/max_speed)
        speed_up_fenzi += exe
    slr_fenmu = min(sum_cp_exe)
    return slr_fenmu, speed_up_fenzi


def cal_slr_cp(set_dag_size, set_dag_ccr, index, pe_nums, pe_types):

    dag_size = set_dag_size
    ccr = set_dag_ccr
    num_index = index

    data_name = f"./dag_data/DAG_{dag_size}_{ccr}_{num_index}.json"

    with open(data_name) as file:
        loaded_dict = json.load(file)
        nodes = loaded_dict['nodes']
        coms = loaded_dict['edges']
    tasks = []
    edges = {}
    for nodde in nodes:
        node = {
            "id": nodde[0],
            "layer_rank": nodde[2],
            "data": nodde[1]
        }
        tasks.append(node)
    for com in coms:
        edges[(com[0][0], com[0][1])] = com[1]

    processors_num = pe_nums
    processors_type = pe_types

    processor_speed = []
    for i in range(processors_type):
        for j in range(int(processors_num/processors_type)):
            processor_speed.append(i+1)

    communication_speed = []
    for i in range(pe_nums):
        temp = [0] * pe_nums
        for j in range(pe_nums):
            if i == j:
                temp[j]= 0
            elif processor_speed[i]==processor_speed[j]:
                temp[j] = 2
            else:
                temp[j] = 1
        communication_speed.append(temp)


    biggest_frequence = max(processor_speed)
    sum_wcet = 0
    for task in tasks:
        sum_wcet += (task['data'] / biggest_frequence)

    parameters = {
        'tasks': tasks,
        'edges': edges,
        'exe_speeds': processor_speed,
        'com_speeds': communication_speed
    }

    slr_fenmu, speedup_fenzi = claculation_algo(**parameters)
    return slr_fenmu, speedup_fenzi


# if __name__ == '__main__':
#     result = cal_slr_cp(60, 1.0, 0, 8, 2)

