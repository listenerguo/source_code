
from GTDE_ALLP import groupTheory as gtde
from HHG_ALLP import hhg
from EGATS_ALLP import egats
from CPOP_ALLP import cpop
from HEFT_ALLP import heft

import json
import pandas as pd
import random



# def func_test():
#     makespan = random.uniform(1,10)
#     exetime = random.uniform(10,20)
#     slr = random.uniform(0.2,1.2)
#     spd = random.uniform(0.1,1.0)
#     return makespan,exetime,slr,spd

# SLR AND HEFT PARAMETER
slr_fenmu_lists = {}
speedup_fenzi_lists ={}
# RESULTS OF ALGORITHMS
    # 1. HEFT
makespan_heft = {}
exetime_heft = {}
slr_heft = {}
spd_heft = {}
    # 2. CPOP
makespan_cpop = {}
exetime_cpop = {}
slr_cpop = {}
spd_cpop = {}
    # 3. GTDE
makespan_gtde = {}
exetime_gtde = {}
slr_gtde = {}
spd_gtde = {}
    # 4. HHG
makespan_hhg = {}
exetime_hhg = {}
slr_hhg = {}
spd_hhg = {}
    # 5. EGATS
makespan_egats = {}
exetime_egats = {}
slr_egats = {}
spd_egats = {}

# EXPERIMENT
dag_sizes = [10, 20, 40, 60, 80, 100]
processors = [4, 8, 12, 16]
cluster_sets = {4: [2], 8: [2, 4], 12: [2, 3, 4, 6], 16: [2, 4, 8]}
dag_types = ['DAG']
dag_ccrs = [0.1, 0.5, 1.0, 2.0, 5.0]
dag_indexs = 0
repeat = 20

for core_num in processors:
    slr_fenmu_lists[core_num]=[]
    speedup_fenzi_lists[core_num] = []

    makespan_heft[core_num] = []
    exetime_heft[core_num] = []
    slr_heft[core_num] = []
    spd_heft[core_num] = []

    makespan_cpop[core_num] = []
    exetime_cpop[core_num] = []
    slr_cpop[core_num] = []
    spd_cpop[core_num] = []

    makespan_gtde[core_num] = []
    exetime_gtde[core_num] = []
    slr_gtde[core_num] = []
    spd_gtde[core_num] = []

    makespan_hhg[core_num] = []
    exetime_hhg[core_num] = []
    slr_hhg[core_num] = []
    spd_hhg[core_num] = []

    makespan_egats[core_num] = []
    exetime_egats[core_num] = []
    slr_egats[core_num] = []
    spd_egats[core_num] = []

    # core_num
    clus = cluster_sets[core_num]
    for gtype in dag_types:
        for num in dag_sizes:
            for ccr in dag_ccrs:
                # 对每个CCR取均值
                temp_heft_makespan = []
                temp_heft_exetime = []
                temp_heft_slr = []
                temp_heft_spd = []

                temp_cpop_makespan = []
                temp_cpop_exetime = []
                temp_cpop_slr = []
                temp_cpop_spd = []

                temp_gtde_makespan = []
                temp_gtde_exetime = []
                temp_gtde_slr = []
                temp_gtde_spd = []

                temp_hhg_makespan = []
                temp_hhg_exetime = []
                temp_hhg_slr = []
                temp_hhg_spd = []

                temp_egats_makespan = []
                temp_egats_exetime = []
                temp_egats_slr = []
                temp_egats_spd = []

                for id in dag_indexs:
                    print(f" ***************  for {gtype}_{num}_{ccr}_{id}.json  *************** ")

                    clu = random.choice(clus)
                    core_speeds = []
                    for a in range(clu):
                        sp = round(random.uniform(1, 2), 1)
                        for b in range(int(core_num / clu)):
                            core_speeds.append(sp)

                        # 1. HEFT
                    m_heft, t_heft, sl_heft, sp_heft = heft(gtype, num, ccr, core_num, clu, core_speeds, id)
                    temp_heft_makespan.append(m_heft)
                    temp_heft_exetime.append(t_heft)
                    temp_heft_slr.append(sl_heft)
                    temp_heft_spd.append(sp_heft)
                        # 2. CPOP
                    m_cpop, t_cpop, sl_cpop, sp_cpop = cpop(gtype, num, ccr, core_num, clu, core_speeds, id)
                    temp_cpop_makespan.append(m_cpop)
                    temp_cpop_exetime.append(t_cpop)
                    temp_cpop_slr.append(sl_cpop)
                    temp_cpop_spd.append(sp_cpop)

                        # 3. GTDE
                    temp_m_lists1 = []
                    temp_t_lists1 = []
                    temp_sl_list1 = []
                    temp_sp_list1 = []
                    for k in range(repeat):
                        m_gtde, t_gtde, sl_gtde, sp_gtde = gtde(gtype, num, ccr, core_num, clu, core_speeds, id)
                        temp_m_lists1.append(m_gtde)
                        temp_t_lists1.append(t_gtde)
                        temp_sl_list1.append(sl_gtde)
                        temp_sp_list1.append(sp_gtde)
                    avg_m_gtde = round(sum(temp_m_lists1) / repeat, 2)
                    avg_t_gtde = round(sum(temp_t_lists1) / repeat, 2)
                    avg_sl_gtde = round(sum(temp_sl_list1) / repeat, 2)
                    avg_sp_gtde = round(sum(temp_sp_list1) / repeat, 2)
                    temp_gtde_makespan.append(avg_m_gtde)
                    temp_gtde_exetime.append(avg_t_gtde)
                    temp_gtde_slr.append(avg_sl_gtde)
                    temp_gtde_spd.append(avg_sp_gtde)
                        # 4. HHG
                    temp_m_lists2 = []
                    temp_t_lists2 = []
                    temp_sl_list2 = []
                    temp_sp_list2 = []
                    for k in range(repeat):
                        m_hhg, t_hhg, sl_hhg, sp_hhg = hhg(gtype, num, ccr, core_num, clu, core_speeds, id)    # func_test()#
                        temp_m_lists2.append(m_hhg)
                        temp_t_lists2.append(t_hhg)
                        temp_sl_list2.append(sl_hhg)
                        temp_sp_list2.append(sp_hhg)
                    avg_m_hhg = round(sum(temp_m_lists2) / repeat, 2)
                    avg_t_hhg = round(sum(temp_t_lists2) / repeat, 2)
                    avg_sl_hhg = round(sum(temp_sl_list2) / repeat, 2)
                    avg_sp_hhg = round(sum(temp_sp_list2) / repeat, 2)
                    temp_hhg_makespan.append(avg_m_hhg)
                    temp_hhg_exetime.append(avg_t_hhg)
                    temp_hhg_slr.append(avg_sl_hhg)
                    temp_hhg_spd.append(avg_sp_hhg)
                        # 5. EGATS
                    temp_m_lists3 = []
                    temp_t_lists3 = []
                    temp_sl_list3 = []
                    temp_sp_list3 = []
                    for k in range(repeat):
                        m_egats, t_egats, sl_egats, sp_egats = egats(gtype, num, ccr, core_num, clu, core_speeds, id)
                        temp_m_lists3.append(m_egats)
                        temp_t_lists3.append(t_egats)
                        temp_sl_list3.append(sl_egats)
                        temp_sp_list3.append(sp_egats)
                    avg_m_ega = round(sum(temp_m_lists3) / repeat, 2)
                    avg_t_ega = round(sum(temp_t_lists3) / repeat, 2)
                    avg_sl_ega = round(sum(temp_sl_list3) / repeat, 2)
                    avg_sp_ega = round(sum(temp_sp_list3) / repeat, 2)
                    temp_egats_makespan.append(avg_m_ega)
                    temp_egats_exetime.append(avg_t_ega)
                    temp_egats_slr.append(avg_sl_ega)
                    temp_egats_spd.append(avg_sp_ega)

                rpt = len(dag_indexs)
                makespan_heft[core_num].append(sum(temp_heft_makespan)/rpt)
                exetime_heft[core_num].append(sum(temp_heft_exetime)/rpt)
                slr_heft[core_num].append(sum(temp_heft_slr)/rpt)
                spd_heft[core_num].append(sum(temp_heft_spd)/rpt)

                makespan_cpop[core_num].append(sum(temp_cpop_makespan)/rpt)
                exetime_cpop[core_num].append(sum(temp_cpop_exetime)/rpt)
                slr_cpop[core_num].append(sum(temp_cpop_slr)/rpt)
                spd_cpop[core_num].append(sum(temp_cpop_spd)/rpt)

                makespan_gtde[core_num].append(sum(temp_gtde_makespan)/rpt)
                exetime_gtde[core_num].append(sum(temp_gtde_exetime)/rpt)
                slr_gtde[core_num].append(sum(temp_gtde_slr)/rpt)
                spd_gtde[core_num].append(sum(temp_gtde_spd)/rpt)

                makespan_hhg[core_num].append(sum(temp_hhg_makespan)/rpt)
                exetime_hhg[core_num].append(sum(temp_hhg_exetime)/rpt)
                slr_hhg[core_num].append(sum(temp_hhg_slr)/rpt)
                spd_hhg[core_num].append(sum(temp_hhg_spd)/rpt)

                makespan_egats[core_num].append(sum(temp_egats_makespan)/rpt)
                exetime_egats[core_num].append(sum(temp_egats_exetime)/rpt)
                slr_egats[core_num].append(sum(temp_egats_slr)/rpt)
                spd_egats[core_num].append(sum(temp_egats_spd)/rpt)

tasks = []
for size in dag_sizes:
    for ccr in dag_ccrs:
        temp = f'{size}_{ccr}'
        tasks.append(temp)
print(f'tasks is:  {tasks}')

processor_lists = processors
algorithms = ['HEFT', 'CPOP', 'EGATS', 'HHG', 'GTDE']

print(f'MAKESPAN of HEFT:  {makespan_heft}')

makespan_data = {
    'HEFT': makespan_heft,
    'CPOP': makespan_cpop,
    'EGATS': makespan_egats,
    'HHG': makespan_hhg,
    'GTDE': makespan_gtde
}
data_makespan = {('processor', 'ALGORITHM'): tasks}
for processor in processor_lists:
    for algorithm in algorithms:
        data_makespan[(f"{processor}", algorithm)] = makespan_data[algorithm][processor]

exetime_data = {
    'HEFT': exetime_heft,
    'CPOP': exetime_cpop,
    'EGATS': exetime_egats,
    'HHG': exetime_hhg,
    'GTDE': exetime_gtde
}
data_exetime = {('processor', 'ALGORITHM'): tasks}

for processor in processor_lists:
    for algorithm in algorithms:
        data_exetime[(f"{processor}", algorithm)] = exetime_data[algorithm][processor]

slr_data = {
    'HEFT': slr_heft,
    'CPOP': slr_cpop,
    'EGATS': slr_egats,
    'HHG': slr_hhg,
    'GTDE': slr_gtde
}
data_slr = {('processor', 'ALGORITHM'): tasks}
for processor in processor_lists:
    for algorithm in algorithms:
        data_slr[(f"{processor}", algorithm)] = slr_data[algorithm][processor]

spd_data = {
    'HEFT': spd_heft,
    'CPOP': spd_cpop,
    'EGATS': spd_egats,
    'HHG': spd_hhg,
    'GTDE': spd_gtde
}
data_spd = {('processor', 'ALGORITHM'): tasks}
for processor in processor_lists:
    for algorithm in algorithms:
        data_spd[(f"{processor}", algorithm)] = spd_data[algorithm][processor]

# Convert the second set of data to a DataFrame
df_data_m = pd.DataFrame(data_makespan)
df_data_t = pd.DataFrame(data_exetime)
df_data_slr = pd.DataFrame(data_slr)
df_data_spd = pd.DataFrame(data_spd)


# Combine both dataframes by stacking them vertically
df_combined = pd.concat([df_data_m, df_data_t, df_data_slr, df_data_spd], keys=['MAKESPAN', 'EXETIME', 'SLR', 'SPEEDUP'], axis=0)
# Save the combined DataFrame to an Excel file
file_path_combined = "./NEW_Random_DAG_Data.xlsx"
df_combined.to_excel(file_path_combined)



