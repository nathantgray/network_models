## trying chatgpt suggestion
# import multiprocessing as mp
from time import perf_counter
import distopf as opf
# from opf import LinDistModelQ, cvxpy_solve, cp_obj_loss
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import cvxpy as cp

# Define area information
area_info = {
    'mgc1': {
        # Area connection information
        'up_mgcs': [],
        'up_global_node_id': [1],
        'down_mgcs': ['mgc2', 'mgc3'],
        'down_local_node_id': [41, 42],
        'down_global_node_id': [14, 18],
        # Controller information
        # 'controller_update_period': global_info['controller_update_period'],
        # 'time_end_sim': global_info['time_end_sim'],
        # 'opf_period': global_info['opf_period'],
        # 'scada_scan_period': global_info['scada_scan_period'],
        # 'data_dir': gld_dir/global_info['model_dir']/'area1',
        # 'neighbors': list(device_info.keys()),
        # 'devices': device_info['mgc1'],
        # # ns-3 node number
        # 'ns3_node': 0,
        # # solver parameter
        # 'ro': global_info.get('ro', 10)
    },
    'mgc2': {
        # Area connection information
        'up_mgcs': ['mgc1'],
        'up_global_node_id': [126],
        'down_mgcs': ['mgc4'],
        'down_local_node_id': [18],
        'down_global_node_id': [66],
        # Controller information
        # 'controller_update_period': global_info['controller_update_period'],
        # 'time_end_sim': global_info['time_end_sim'],
        # 'opf_period': global_info['opf_period'],
        # 'scada_scan_period': global_info['scada_scan_period'],
        # 'data_dir': gld_dir/global_info['model_dir']/'area2',
        # 'neighbors': list(device_info.keys()),
        # 'devices': device_info['mgc2'],
        # # ns-3 node number
        # 'ns3_node': 1,
        # # solver parameter
        # 'ro': global_info.get('ro', 10)
    },
    'mgc3': {
        # Area connection information
        'up_mgcs': ['mgc1'],
        'up_global_node_id': [125],
        'down_mgcs': [],
        'down_local_node_id': [],
        'down_global_node_id': [],
        # Controller agent information
        # 'controller_update_period': global_info['controller_update_period'],
        # 'time_end_sim': global_info['time_end_sim'],
        # 'opf_period': global_info['opf_period'],
        # 'data_dir': gld_dir/global_info['model_dir']/'area3',
        # 'neighbors': list(device_info.keys()),
        # 'devices': device_info['mgc3'],
        # # ns-3 node number
        # 'ns3_node': 2,
        # # solver parameter
        # 'ro': global_info.get('ro', 10)
    },
    'mgc4': {
        # Area connection information
        'up_mgcs': ['mgc2'],
        'up_global_node_id': [128],
        'down_mgcs': [],
        'down_local_node_id': [],
        'down_global_node_id': [],
        # Controller information
        # 'controller_update_period': global_info['controller_update_period'],
        # 'time_end_sim': global_info['time_end_sim'],
        # 'opf_period': global_info['opf_period'],
        # 'data_dir': gld_dir/global_info['model_dir']/'area4',
        # 'neighbors': list(device_info.keys()),
        # 'devices': device_info['mgc4'],
        # # ns-3 node number
        # 'ns3_node': 3,
        # # solver parameter
        # 'ro': global_info.get('ro', 10)
    }
}

# Process function for each area
def process_area(area_case, area_name, area2_load_p, area3_load_p, area4_load_p, area2_load_q, area3_load_q, area4_load_q, area2_volt, area3_volt, area4_volt):

    # branch_data = area_case["branch_data"]
    # bus_data = area_case["bus_data"]
    # gen_data = area_case["gen_data"]
    # cap_data = area_case["cap_data"]
    # reg_data = area_case["reg_data"]
    # bus_data.v_max = 1.05
    # loadshape_data = area_case["loadshape_data"]
    # pv_loadshape_data = area_case["pv_loadshape_data"]


    case = opf.DistOPFCase(
        data_path=Path.cwd()/area_name,
        gen_mult=2,
        load_mult=1,
        v_swing=1.0,
        v_max=1.05,
        v_min=0.95,
    )
    # case.gen_data.a_mode = "CONSTANT_P"
    # case.gen_data.b_mode = "CONSTANT_P"
    # case.gen_data.c_mode = "CONSTANT_P"
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    case.gen_data.sa_max = 0.048
    case.gen_data.sb_max = 0.048
    case.gen_data.sc_max = 0.048
    # for t in range(LinDistModelQ.n):

    if area_name == 'area1':
        # print(
        #     f"Area1: Entering solver with s_down={np.c_[(area2_load_p[-1] + 1j*area2_load_q[-1]).flatten(), (area3_load_p[-1] + 1j*area3_load_q[-1]).flatten()].T}")
        case.bus_data.loc[case.bus_data.index[-2], ["pl_a", "pl_b", "pl_c"]] = area2_load_p[-1]
        case.bus_data.loc[case.bus_data.index[-2], ["ql_a", "ql_b", "ql_c"]] = area2_load_q[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["pl_a", "pl_b", "pl_c"]] = area3_load_p[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["ql_a", "ql_b", "ql_c"]] = area3_load_q[-1]
        # for ph in "abc":
        #     case.bus_data.loc[40, f"pl_{ph}"] = area2_load_p[-1]["abc".index(ph)]
        #     case.bus_data.loc[40, f"ql_{ph}"] = area2_load_q[-1]["abc".index(ph)]
        #     case.bus_data.loc[41, f"pl_{ph}"] = area3_load_p[-1]["abc".index(ph)]
        #     case.bus_data.loc[41, f"ql_{ph}"] = area3_load_q[-1]["abc".index(ph)]
    elif area_name == 'area2':
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area2_volt[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["pl_a", "pl_b", "pl_c"]] = area4_load_p[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["ql_a", "ql_b", "ql_c"]] = area4_load_q[-1]
        # for ph in "abc":
        #     case.bus_data.loc[19, f"pl_{ph}"] = area4_load_p[-1]["abc".index(ph)]
        #     case.bus_data.loc[19, f"ql_{ph}"] = area4_load_q[-1]["abc".index(ph)]
    elif area_name == 'area3':
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area3_volt[-1]
    elif area_name == 'area4':
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area4_volt[-1]

    model1 = opf.LinDistModelCapacitorRegulatorMI(
        case.branch_data,
        case.bus_data,
        case.gen_data,
        case.cap_data,
        case.reg_data
    )
    # res = opf.cvxpy_solve(model1, opf.cp_obj_loss)
    res = model1.solve(opf.cp_obj_loss)
    if not res.success:
        raise ValueError(res.message)
    obj_val = res.fun
    v = model1.get_voltages(res.x)
    s = model1.get_apparent_power_flows(res.x)
    if area_name == 'area1':
        print(f"area1: ***Solver returned with v_down={v.iloc[-2:, 1:].to_numpy().astype(float)}, and s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    if area_name == "area2":
        print(f"area2: ***Solver returned with v_down={v.iloc[-1:, 1:].to_numpy().astype(float)}, and s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    if area_name == "area3":
        print(f"area3: ***Solver returned with s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    if area_name == "area4":
        print(f"area4: ***Solver returned with s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    return area_name, v, s, obj_val

# Load case data from files
def load_case_data(area_folders):
    case_data = {}
    base_path = Path.cwd()
    # base_path = r"C:\Users\anup.parajuli\Desktop\pythonProject\opf_tool\examples"
    for area_folder in area_folders:
        area_path = os.path.join(base_path, area_folder)
        case_data[area_folder] = {
            "bus_data": pd.read_csv(os.path.join(area_path, "bus_data.csv")),
            "branch_data": pd.read_csv(os.path.join(area_path, "branch_data.csv")),
            "gen_data": pd.read_csv(os.path.join(area_path, "gen_data.csv")),
            "cap_data": pd.read_csv(os.path.join(area_path, "cap_data.csv")),
            "reg_data": pd.read_csv(os.path.join(area_path, "reg_data.csv")),
        }
    return case_data
def mapping(area_folders):
    map_all = {}
    anti_map_all = {}

    # Iterate through each area folder
    for area in area_folders:
        area_path = Path.cwd()/area
        # area_path = os.path.join(r"C:\Users\anup.parajuli\Desktop\pythonProject\opf_tool\examples", area)
        map_file_path = os.path.join(area_path, "node_map.json")

        # Read JSON file into a dictionary
        map_data = pd.read_json(map_file_path, typ='series').to_dict()

        # Update map_all with the contents of the current map_data
        map_all[area] = map_data

    # Create the reverse mapping
        for key, value in map_all.items():
            anti_map_all[area] = {value: key for key, value in map_all[area].items()}

    return map_all, anti_map_all

def to_global(data, map):
    v_list = []
    s_list = []

    for area in data.keys():
        anti_map[area] = {value: key for key, value in map[area].items()}
        indexes = list(map[area].values())
        v_df = data[area][0].loc[indexes].copy()
        v_df["id"] = v_df.index
        v_df.id = v_df.id.apply(lambda x: anti_map[area][x])
        v_list.append(v_df)
    v = pd.concat(v_list)
    v = v.sort_values(by="id", ascending=True)
    v.index = v.id
    v = v.drop(columns=["id"])
    for area in data.keys():
        anti_map[area] = {value: key for key, value in map[area].items()}
        indexes = list(map[area].values())
        s_df = data[area][1].loc[indexes[1:]].copy()
        s_df["id"] = s_df.index
        s_df.id = s_df.id.apply(lambda x: anti_map[area][x])
        s_list.append(s_df)
    s = pd.concat(s_list)
    s = s.sort_values(by="id", ascending=True)
    s.index = s.id
    s = s.drop(columns=["id"])
    return v, s

def orig_index(antimap, data):
    for outer_key, outer_dict in data.items():
        for area, mapping in antimap.items():
            df = outer_dict[area]
            df.index = df.index.map(mapping)
            outer_dict[area] = df
        concatenated_df = pd.concat(outer_dict.values())
        sorted_df = concatenated_df.sort_index()
        data[outer_key] = sorted_df
    return data



# Main execution
if __name__ == "__main__":
    tic = perf_counter()
    area_folders = ['area1', 'area2', 'area3', 'area4']
    case_data = load_case_data(area_folders)

    # Initialize shared variables for the iteration
    area2_volt = [np.array([1.0004801,  1.02718284, 1.01085467])]
    area3_volt = [np.array([0.99188649, 1.02537392, 1.00589861])]
    area4_volt = [np.array([0.97579233, 1.01219571, 0.99138788])]
    area2_load = np.array([(448037.1070906019 + 431771.2325436381j), (324178.8650423707 + 347992.7603066683j), (322713.5624203948 + 356098.0150280295j)])/1e6
    area3_load = np.array([(180901.95374716038 + 181335.06431763733j), (155429.36004833697 + 176204.46190488304j), (80268.36904842434 + 125585.10048596119j)])/1e6
    area4_load = np.array([(367104.6358292276+312849.62124166597j), (250809.61971055262+241729.39613609755j), (251388.1085463847+241802.9732501536j)])/1e6
    area2_load_p = [area2_load.real]
    area3_load_p = [area3_load.real]
    area4_load_p = [area4_load.real]
    area2_load_q = [area2_load.imag]
    area3_load_q = [area3_load.imag]
    area4_load_q = [area4_load.imag]

    # area2_load_p, area3_load_p, area4_load_p = [np.zeros(3)], [np.zeros(3)], [np.zeros(3)]
    # area2_load_q, area3_load_q, area4_load_q = [np.zeros(3)], [np.zeros(3)], [np.zeros(3)]
    # area2_volt, area3_volt, area4_volt = [np.ones(3)], [np.ones(3)], [np.ones(3)]

    # pool = mp.Pool(processes=len(area_folders))
    data = {}
    for it in range(50):  # Set a max number of iterations
        print(it)
        # results = pool.starmap(process_area, [(case_data[area], area, area2_load_p, area3_load_p, area4_load_p,
        #                                        area2_load_q, area3_load_q, area4_load_q, area2_volt, area3_volt,
        #                                        area4_volt) for area in area_folders])
        # data = {area_name: (v, s, obj_val) for area_name, v, s, obj_val in results}

        result_area1 = process_area(case_data["area1"], "area1", area2_load_p, area3_load_p, area4_load_p, area2_load_q, area3_load_q, area4_load_q, area2_volt, area3_volt, area4_volt)
        result_area2 = process_area(case_data["area2"], "area2", area2_load_p, area3_load_p, area4_load_p, area2_load_q, area3_load_q, area4_load_q, area2_volt, area3_volt, area4_volt)
        result_area3 = process_area(case_data["area3"], "area3", area2_load_p, area3_load_p, area4_load_p, area2_load_q, area3_load_q, area4_load_q, area2_volt, area3_volt, area4_volt)
        result_area4 = process_area(case_data["area4"], "area4", area2_load_p, area3_load_p, area4_load_p, area2_load_q, area3_load_q, area4_load_q, area2_volt, area3_volt, area4_volt)
        data = {
            "area1": result_area1[1:],
            "area2": result_area2[1:],
            "area3": result_area3[1:],
            "area4": result_area4[1:]
        }

        # Calculate new shared variables based on the results
        new_area2_load = data['area2'][1].loc[data['area2'][1]['fb'] == "up", ["a", "b", "c"]].to_numpy()
        new_area3_load = data['area3'][1].loc[data['area3'][1]['fb'] == "up", ["a", "b", "c"]].to_numpy()
        new_area4_load = data['area4'][1].loc[data['area4'][1]['fb'] == "up", ["a", "b", "c"]].to_numpy()

        # Update the shared variables with the latest results
        area2_load_p.append(new_area2_load.real.copy())
        area3_load_p.append(new_area3_load.real.copy())
        area4_load_p.append(new_area4_load.real.copy())
        area2_load_q.append(new_area2_load.imag.copy())
        area3_load_q.append(new_area3_load.imag.copy())
        area4_load_q.append(new_area4_load.imag.copy())

        # Voltage updates
        area2_volt_v = data["area1"][0].iloc[-2, -3:].to_numpy()
        area3_volt_v = data["area1"][0].iloc[-1, -3:].to_numpy()
        area4_volt_v = data["area2"][0].iloc[-1, -3:].to_numpy()
        # area2_volt_v = np.vstack([data['area1'][0][key].loc[data['area1'][0][key].index == 38, ["a", "b", "c"]].to_numpy() for key in data['area1'][0].keys()])
        # area3_volt_v = np.vstack([data['area1'][0][key].loc[data['area1'][0][key].index == 39, ["a", "b", "c"]].to_numpy() for key in data['area1'][0].keys()])
        # area4_volt_v = np.vstack([data['area2'][0][key].loc[data['area2'][0][key].index == 19, ["a", "b", "c"]].to_numpy() for key in data['area2'][0].keys()])

        area2_volt.append(area2_volt_v.copy())
        area3_volt.append(area3_volt_v.copy())
        area4_volt.append(area4_volt_v.copy())

        # Convergence check (skip the first iteration)
        if it > 0:
            max_diff = max(
                np.max(abs(area2_load_p[-1] - area2_load_p[-2])),
                np.max(abs(area3_load_p[-1] - area3_load_p[-2])),
                np.max(abs(area4_load_p[-1] - area4_load_p[-2])),
                np.max(abs(area2_load_q[-1] - area2_load_q[-2])),
                np.max(abs(area3_load_q[-1] - area3_load_q[-2])),
                np.max(abs(area4_load_q[-1] - area4_load_q[-2])),
                np.max(abs(area2_volt[-1] - area2_volt[-2])),
                np.max(abs(area3_volt[-1] - area3_volt[-2])),
                np.max(abs(area4_volt[-1] - area4_volt[-2]))
            )
            print(max_diff)
            # print(area4_load_q[-3:])

            if max_diff < 1e-3:
                print(f"Converged after {it} iterations")
                print(f"total objective value for DOPF:{sum([data[area][2] for area in area_folders])}")
                break

    # pool.close()
    # pool.join()

    print(f"Time taken for DOPF: {perf_counter() - tic} seconds")

    # Post-process and plot results
    v_all = {}
    for area in area_folders:
        v_all[area] = data[area][0]
    map, anti_map = mapping(area_folders)
    v_dopf, s_dopf = to_global(data, map)


    ## Solvng COPF
    tic1 = perf_counter()

    case = opf.DistOPFCase(
        data_path=Path.cwd(),
        gen_mult=2,
        load_mult=1,
        v_swing=1.0,
        v_max=1.05,
        v_min=0.95,
    )
    # case.gen_data.a_mode = "CONSTANT_P"
    # case.gen_data.b_mode = "CONSTANT_P"
    # case.gen_data.c_mode = "CONSTANT_P"
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    case.gen_data.sa_max = 0.048
    case.gen_data.sb_max = 0.048
    case.gen_data.sc_max = 0.048
    model = opf.LinDistModelCapacitorRegulatorMI(
        case.branch_data,
        case.bus_data,
        case.gen_data,
        case.cap_data,
        case.reg_data
    )

    res = model.solve(opf.cp_obj_loss)
    # res = opf.cvxpy_solve(model, opf.cp_obj_loss)
    print(f"Time taken for COPF: {perf_counter() - tic1} seconds")
    print(f"objective value from COPF: {res.fun}")
    v = model.get_voltages(res.x)
    s = model.get_apparent_power_flows(res.x)
    v_diff = v.copy()
    v_diff.loc[:, ["a", "b", "c"]] = v.loc[:, ["a", "b", "c"]] - v_dopf.loc[:, ["a", "b", "c"]]
    print(v_diff.loc[:, ["a", "b", "c"]].max())
    s_diff = s.copy()
    s_diff.loc[:, ["a", "b", "c"]] = s.loc[:, ["a", "b", "c"]] - s_dopf.loc[:, ["a", "b", "c"]]
    print(s_diff.loc[:, ["a", "b", "c"]].apply(np.real).max())
    print(s_diff.loc[:, ["a", "b", "c"]].apply(np.imag).max())
    # sorted_v_df = pd.concat(sorted_v.values(), keys=sorted_v.keys())
    # sorted_v_df = sorted_v_df.reset_index()
    # sorted_v_df.columns = ['time', 'node', 'a', 'b', 'c']
    # sorted_v_df.to_csv("DOPF.csv", index=False)
    # v_df = pd.concat(v.values(),keys=v.keys())
    # v_df = v_df.reset_index()
    # v_df.columns = ['time','node','a','b','c']
    # v_df.to_csv("COPF.csv",index = False)
