# Required Libraries
import copy
import math
import numpy as np
import random
import os
import time
import json
import pympsym
import matplotlib.pyplot as plt
import networkx as nx

from calculator_slr_spd import cal_slr_cp

def compute_ud_rank(graph, communication_time, computation_time):
    task_rank = []
    task_number = len(graph)
    # downward_rank
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
            if graph[i]["rank"] - max_value <= 0.1 and cp_lists[-1] in graph[i]["predecessors"]:
                cp_lists.append(i)
                break
    return cp_lists


def initial_population(tasks, prcess_num, size_pop, process_speed, graph, com_time, exe_time):
    rand_pop = []
    oppo_pop = []
    choise_list = list(range(prcess_num))

    for i in range(size_pop-10):
        map = []
        op_m = []
        for i in range(len(tasks)):
            element = random.choices(choise_list, process_speed)[0]
            map.append(element)
            value = int((prcess_num - 1) - map[-1])
            if value < 0:
                value = random.choice(choise_list)
            op_m.append(value)
        set_pop = set(map)
        if len(set_pop) ==1:
            map = random.choices(choise_list, process_speed, k=len(tasks))
            op_m = [prcess_num-i for i in map]
        rand_pop.append(map)
        oppo_pop.append(op_m)
    init_pop = rand_pop + oppo_pop

    max_proc = process_speed.index(max(process_speed))
    cp_task_list = compute_ud_rank(graph, com_time, exe_time)

    for i in range(18):
        give_pop = []
        for i in range(len(tasks)):
            if i in cp_task_list and len(cp_task_list) != len(tasks):
                give_pop.append(max_proc)
            else:
                give_pop.append(random.choices(range(len(process_speed)), process_speed)[0])
        init_pop.append(give_pop)
    give_pop = []
    for i in range(len(tasks)):
        if i in cp_task_list:
            give_pop.append(max_proc)
        else:
            give_pop.append(i % prcess_num)
    init_pop.append(give_pop)
    give_pop = [max_proc] * len(tasks)
    init_pop.append(give_pop)
    return init_pop


def classify_function(ag, repre, population, classify_dic):
    cla_pop = []
    cla_rep = []
    old_pop = []
    old_fitness = []
    for mapping in population:
        rep_map, new, _ = ag.representative(mapping, repre, method='local_search_dfs')
        if new:
            cla_pop.append(mapping)
            cla_rep.append(rep_map)
        else:
            if rep_map not in cla_rep:
                old_pop.append(mapping)
                old_fitness.append(classify_dic[rep_map])
    return cla_pop, cla_rep, old_pop, old_fitness

def fitness_function(rest_pop, rest_rep, graph, tasks, edges, task_exe_time, com_speeds, classify_dic):
    fitness = []
    for mapping in rest_pop:
        exe_time = []
        for task, processor in enumerate(mapping):
            exe_time.append(task_exe_time[task][processor])
        com_time = {}
        for edge,com_data in edges.items():
            task_i, task_j = edge[0], edge[1]

            p_i = mapping[task_i]
            p_j = mapping[task_j]

            if p_i == p_j:
                com_time[edge] = 0
            else:
                speed = com_speeds[p_i][p_j]
                com_time[edge] = round(com_data/speed, 1) if speed != 0 else float('inf')
        graph_temp = copy.deepcopy(graph)
        task_ranks = compute_rank(graph_temp, com_time, exe_time)
        task_priority_dict = dict(task_ranks)
        task_lists = [x[0] for x in task_ranks]

        num_of_pe = len(com_speeds)
        process_exe = []
        for i in range(num_of_pe):
            process_exe.append([(0, 0)])

        bus = [(0, 0)]
        finish_task = [-1] * len(mapping)
        true_mapping = {}
        for i in range(len(mapping)):
            true_mapping[task_lists[i]] = mapping[i]

        ready_lists = [task_lists[0]]
        finished_lists = []
        waiting_lists = task_lists[1:]
        schedule_run = True
        while schedule_run:
            for t in waiting_lists:
                if len(graph[t]["predecessors"]) > 0 and all(
                        task in finished_lists for task in graph[t]["predecessors"]):
                    ready_lists.append(t)
                    waiting_lists.remove(t)
            if len(waiting_lists) == 0:
                schedule_run = False
            ready_tasks_with_priorities = [(task_id, task_priority_dict[task_id]) for task_id in ready_lists]
            task = max(ready_tasks_with_priorities, key=lambda x: x[1])[0]
            process = true_mapping[task]
            graph_temp[task]["mapping"] = process

            bus = sorted(bus)
            last_com = bus[-1][-1]
            last_exe = []
            for i in process_exe:
                last_exe.append(max(max(i)))
            eft = 0
            temp_com = [(0, 0)]
            if task == 0:
                eft = exe_time[task]
            else:
                pred_finished = []
                for pred in graph_temp[task]["predecessors"]:
                    pred_finished.append([pred, graph_temp[pred]['end_time']])
                pred_finished = sorted(pred_finished, key=lambda x: x[1])

                max_pred_end = last_com
                for pred, finished_time in pred_finished:
                    pred_map = graph_temp[pred]["mapping"]
                    if pred_map != process:
                        cs_start = max(max_pred_end, finished_time)
                        com_time = round(edges[(pred, task)] / com_speeds[pred_map][process], 1)
                        cs_finish = cs_start + com_time
                        temp_com.append((cs_start,cs_finish))
                        max_pred_end = cs_finish
                exe = exe_time[task]
                task_start = max(last_exe[process], max_pred_end)
                eft = task_start + exe
            graph_temp[task]['end_time'] = eft
            graph_temp[task]['start_time'] = eft - exe_time[task]
            process_exe[process].append((graph_temp[task]['start_time'], graph_temp[task]['end_time']))
            process_exe[process] = sorted(process_exe[process])
            for i in temp_com:
                bus.append(i)
            bus = list(set(bus))
            bus = sorted(bus)
            ready_lists.remove(task)
            finished_lists.append(task)

        makespan = 0
        for id in range(len(graph_temp)):
            if graph_temp[id]["end_time"] > makespan:
                makespan = graph_temp[id]["end_time"]
        fitness.append(makespan)
        index = rest_pop.index(mapping)
        classify_dic[rest_rep[index]] = makespan
    return fitness


def selection_function(rest_pop, rest_fit, old_pop, old_fit, size_pop, iter, max_iteration):
    if iter/max_iteration < 0.5:
        need_o = round(size_pop * 0.15)
    else:
        need_o = round(size_pop * 0.8)
    need_r = size_pop - need_o
    pop_now = []
    fitness = []
    if len(old_pop) >= need_o and len(rest_pop) >= need_r:
        sort_old_id = sorted(range(len(old_fit)), key=lambda x: old_fit[x])
        need_old_size = need_o
        sort_rest_id = sorted(range(len(rest_fit)), key=lambda x: rest_fit[x])
        need_rest_size = need_r
        for i in sort_rest_id[:need_rest_size]:
            pop_now.append(rest_pop[i])
            fitness.append(rest_fit[i])
        for i in sort_old_id[:need_old_size]:
            pop_now.append(old_pop[i])
            fitness.append(old_fit[i])
    else:
        if len(old_pop) < need_o:
            need_old_size = len(old_pop)
            for i in range(need_old_size):
                pop_now.append(old_pop[i])
                fitness.append(old_fit[i])
            if len(rest_pop) >= size_pop-len(old_pop):
                need_rest_size = size_pop-need_old_size
                sort_rest_id = sorted(range(len(rest_fit)), key=lambda x: rest_fit[x])
                for i in sort_rest_id[:need_rest_size]:
                    pop_now.append(rest_pop[i])
                    fitness.append(rest_fit[i])

        if len(rest_pop) < need_r:
            need_rest_size = len(rest_pop)
            for i in range(need_rest_size):
                pop_now.append(rest_pop[i])
                fitness.append(rest_fit[i])
            if len(old_pop) >= size_pop-len(rest_pop):
                need_old_size = size_pop - need_rest_size
                sort_old_id = sorted(range(len(old_fit)), key=lambda x: old_fit[x])
                for i in sort_old_id[:need_old_size]:
                    pop_now.append(old_pop[i])
                    fitness.append(old_fit[i])

    return pop_now, fitness


def cross_mutation(population, best_global, f, cr, process_num):
    min_values = 0
    max_values = process_num - 1
    cm_pop = []
    for i in range(len(population)):
        choise_list = list(range(len(population)))
        choise_list.remove(i)
        r1 = random.choice(choise_list)
        choise_list.remove(r1)
        r2 = random.choice(choise_list)
        choise_list.remove(r2)
        r3 = random.choice(choise_list)
        choise_list.remove(r3)
        r4 = random.choice(choise_list)
        choise_list.remove(r4)
        r5 = random.choice(choise_list)
        rand_j = random.randint(0, len(best_global))
        v = copy.deepcopy(best_global)
        for j in range(len(best_global)):
            rand_p = random.random()
            mutation_prob = j % 4
            if rand_p <= cr or j == rand_j:
                if mutation_prob == 0:
                    v[j] = round(population[r3][j] + f * (population[r1][j] - population[r2][j]))
                elif mutation_prob == 1:
                    v[j] = round(best_global[j] + f * (population[r1][j] - population[r2][j]))
                elif mutation_prob == 2:
                    v[j] = round(population[r3][j] + f * (population[r1][j] - population[r2][j]) + f * (population[r4][j] - population[r5][j]))
                else:
                    v[j] = round(population[r3][j] + f * (best_global[j] - population[r3][j])+ f * (population[r1][j] - population[r2][j]))
            else:
                v[j] = population[i][j]
            if v[j] >= max_values:
                v[j] = max_values
            elif v[j] < min_values:
                v[j] = min_values
        cm_pop.append(v)
    return cm_pop

def initial_heft(graph, tasks , edges, mean_c , mean_w, exe_speeds, com_speeds):
    task_rank_u = compute_rank(graph, mean_c, mean_w)
    task_priority_dict = dict(task_rank_u)
    task_lists = [x[0] for x in task_rank_u]

    mappings = [-1] * len(tasks)
    num_of_pe = len(exe_speeds)
    process_exe = []
    for i in range(num_of_pe):
        process_exe.append([(0, 0)])

    bus = [(0, 0)]
    ready_lists = [task_lists[0]]
    finished_lists = []
    waiting_lists = task_lists[1:]
    schedule_run = True
    while schedule_run:
        for t in waiting_lists:
            if len(graph[t]["predecessors"]) > 0 and all(task in finished_lists for task in graph[t]["predecessors"]):
                ready_lists.append(t)
                waiting_lists.remove(t)
        if len(waiting_lists) == 0:
            schedule_run = False
        ready_tasks_with_priorities = [(task_id, task_priority_dict[task_id]) for task_id in ready_lists]
        task = max(ready_tasks_with_priorities, key=lambda x: x[1])[0]

        bus = sorted(bus)
        last_com = bus[-1][-1]
        last_exe = []
        for i in process_exe:
            last_exe.append(max(max(i)))
        eft = [0] * num_of_pe
        temp_com = []
        for process in range(num_of_pe):
            temp_com.append([(0, 0)])
        for p in range(num_of_pe):
            if task == 0:
                eft[p] = round(tasks[task]["data"] / exe_speeds[p], 1)
            else:

                pred_finished = []
                for pred in graph[task]["predecessors"]:
                    pred_finished.append([pred, graph[pred]['end_time']])
                pred_finished = sorted(pred_finished, key=lambda x: x[1])

                max_pred_end = last_com
                for pred, finished_time in pred_finished:
                    pred_map = graph[pred]["mapping"]
                    if pred_map != p:
                        cs_start = max(max_pred_end, finished_time)
                        com_time = round(edges[(pred, task)] / com_speeds[pred_map][p], 1)
                        cs_finish = cs_start + com_time
                        temp_com[p].append((cs_start, cs_finish))
                        max_pred_end = cs_finish
                exe = round(tasks[task]["data"] / exe_speeds[p], 1)
                task_start = max(last_exe[p], max_pred_end)
                eft[p] = task_start + exe

        map = eft.index(min(eft))
        mappings[task] = map
        graph[task]['end_time'] = eft[map]
        graph[task]['start_time'] = eft[map] - round(tasks[task]["data"] / exe_speeds[map], 1)
        graph[task]['mapping'] = map
        process_exe[map].append((graph[task]['start_time'], graph[task]['end_time']))
        process_exe[map] = sorted(process_exe[map])

        for i in temp_com[map]:
            bus.append(i)

        bus = list(set(bus))
        bus = sorted(bus)

        ready_lists.remove(task)
        finished_lists.append(task)
    return mappings


def group_de_algo(archgraph, representatives, tasks, edges, exe_speeds, com_speeds, size_population, max_iter):

    ag = archgraph
    repre = representatives

    pe_num = len(exe_speeds)
    task_num = len(tasks)
    task_exe_time = []
    for task in tasks:
        temp = []
        for s in exe_speeds:
            temp.append(round(task['data']/s, 1))
        task_exe_time.append(temp)

    mean_w = []
    for task_in_one_pro in task_exe_time:
        sum_exe = sum(task_in_one_pro)
        mean_w.append(sum_exe/len(exe_speeds))
    mean_c = {}
    sum_com_speed = 0
    for speed in com_speeds:
        sum_com_speed += sum(speed)
    for edge, data in edges.items():
        mean_c[edge] = round(data/sum_com_speed, 1)


    graph = {}
    for t_id in range(task_num):
        graph[t_id] = {"id": tasks[t_id]['id'], "layer_level": tasks[t_id]['layer_rank'], "upward_rank": 0,
                       "downward_rank": 0, 'rank': 0,
                       "successors": [], "predecessors": [], "start_time": None, "end_time": 0, "mapping": None}

    for edge, _ in edges.items():
        predecessor = edge[0]
        successor = edge[1]
        graph[predecessor]["successors"].append(successor)
        graph[successor]["predecessors"].append(predecessor)

    init_pop = initial_population(tasks, pe_num, size_population, exe_speeds, graph, mean_c, mean_w)

    temp_graph = copy.deepcopy(graph)
    heft_map = initial_heft(graph, tasks, edges, mean_c, mean_w, exe_speeds, com_speeds)
    init_pop[0] = heft_map

    iter = 0
    classify = {}
    max_stop = 50
    history = []

    while True:
        if iter == 0:
            population = init_pop
        else:
            population = new_pop

        rest_pop, rest_rep, old_pop, old_fit = classify_function(ag, repre, population, classify)
        rest_fit = fitness_function(rest_pop, rest_rep, graph, tasks, edges, task_exe_time, com_speeds, classify)
        pop_now, fitness = selection_function(rest_pop, rest_fit, old_pop, old_fit, size_population, iter, max_iter)
        sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness)
        range_best = round(0.1*size_population)
        good_mapping_id = random.choice(sorted_indices[:range_best])
        good_mapping = pop_now[good_mapping_id]
        min_fit = min(fitness)
        best_mapping = pop_now[fitness.index(min_fit)]

        if iter == 0:
            makespan = min_fit
        if makespan > min_fit:
            makespan = min_fit
        history.append(min_fit)
        if len(history) > 100:
            last_results = history[-max_stop:]
            if all(result == last_results[0] for result in last_results):
                break

        if iter < max_iter:
            pc = 0.5
            pm = 1/len(tasks)
            cm_pop = cross_mutation(pop_now, good_mapping, pm, pc, pe_num)
            new_pop = pop_now + cm_pop
        else:
            break
        iter += 1
    return makespan

def groupTheory(set_dag_type, set_dag_size, set_dag_ccr, pe_nums, pe_cluster, core_speeds, dag_id=None):
    """
        set_dag_type:   DAG type: DAG
        set_dag_size:   10   {10, 20, 40, 60, 80, 100}
        set_dag_ccr:    0.1  {0.1, 0.5, 1.0, 2.0, 5.0}
        pe_nums:        8
        pe_cluster:     2
        core_speeds：   [1.2, 1.2, 1.2, 1.2, 1.9, 1.9,...]
        dag_id:        （= None）
    """

    dag_type = set_dag_type
    dag_size = set_dag_size
    ccr = set_dag_ccr
    num_index = dag_id

    if dag_type == "DAG":
        data_name = f"./dag_data/{dag_type}_{dag_size}_{ccr}_{num_index}.json"
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
    else:
        print('ERROR')
        return None

    processors_num = pe_nums
    cluster_num = pe_cluster
    processor_speed = core_speeds
    communication_speed = []
    for i in range(pe_nums):
        temp = [0] * pe_nums
        for j in range(pe_nums):
            if i == j:
                temp[j] = 0
            elif processor_speed[i] == processor_speed[j]:
                temp[j] = 1.5
            else:
                temp[j] = 1
        communication_speed.append(temp)

    slr_fenmu, speedup_fenzi = cal_slr_cp(set_dag_type, set_dag_size, ccr, processors_num, processor_speed,
                                          communication_speed, dag_id)


    ag = pympsym.ArchGraph(directed = False)
    for i in range(cluster_num):
        num = int(processors_num/cluster_num)
        ag.add_processors(num,f'P{i}')
    for i in range(processors_num):
        for j in range(i+1, processors_num):
            if processor_speed[i] == processor_speed[j]:
                ag.add_channel(i, j, 'C1')
            else:
                ag.add_channel(i, j, 'C2')
    ag.initialize()


    size_pop = 200
    max_iter = 200
    parameters = {
        'tasks': tasks,
        'edges': edges,
        'exe_speeds': processor_speed,
        'com_speeds': communication_speed,
        'size_population': size_pop,
        'max_iter': max_iter
    }

    representatives = pympsym.Representatives()
    start_time = time.time()
    makespan = group_de_algo(ag, representatives, **parameters)
    end_time = time.time()
    execution_time = end_time - start_time

    slr = makespan / slr_fenmu
    speedup = speedup_fenzi / makespan
    print("GTDE：T-", execution_time, "M-", makespan, " SLR-", slr, " SPEEDUP-", speedup)
    return makespan, execution_time, slr, speedup


# if __name__ == '__main__':
#     groupTheory('DAG',40,1.0,4,2,[1.2, 1.2, 1.6, 1.6],2)

