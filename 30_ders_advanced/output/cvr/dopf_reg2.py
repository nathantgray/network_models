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
    case.bus_data.cvr_p = 0.6
    case.bus_data.cvr_q = 1.5
    # case.bus_data.bus_type = opf.SWING_FREE
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
        # case.bus_data.at[case.bus_data.index[-2], "bus_type"] = opf.PQ_FREE
        # case.bus_data.at[case.bus_data.index[-1], "bus_type"] = opf.PQ_FREE
        case.bus_data.loc[case.bus_data.index[-2], ["pl_a", "pl_b", "pl_c"]] = area2_load_p[-1]
        case.bus_data.loc[case.bus_data.index[-2], ["ql_a", "ql_b", "ql_c"]] = area2_load_q[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["pl_a", "pl_b", "pl_c"]] = area3_load_p[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["ql_a", "ql_b", "ql_c"]] = area3_load_q[-1]
        # case.bus_data.loc[case.bus_data.index[-2], ["v_a", "v_b", "v_c"]] = area2_volt[-1]
        # case.bus_data.loc[case.bus_data.index[-1], ["v_a", "v_b", "v_c"]] = area3_volt[-1]
    elif area_name == 'area2':
        # case.bus_data.at[0, "bus_type"] = opf.SWING_FREE
        # case.bus_data.at[case.bus_data.index[-1], "bus_type"] = opf.PQ_FREE
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area2_volt[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["pl_a", "pl_b", "pl_c"]] = area4_load_p[-1]
        case.bus_data.loc[case.bus_data.index[-1], ["ql_a", "ql_b", "ql_c"]] = area4_load_q[-1]
        # case.bus_data.loc[case.bus_data.index[-1], ["v_a", "v_b", "v_c"]] = area4_volt[-1]
    elif area_name == 'area3':
        # case.bus_data.at[0, "bus_type"] = opf.SWING_FREE
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area3_volt[-1]
    elif area_name == 'area4':
        # case.bus_data.at[0, "bus_type"] = opf.SWING_FREE
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area4_volt[-1]

    # model1 = opf.LinDistModelPQ(
    model1 = opf.LinDistModelCapacitorRegulatorMI(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data
    )
    # res = opf.cvxpy_solve(model1, opf.cp_obj_loss)
    # res = model1.solve(cp_obj_loss_relaxed)
    res = model1.solve(opf.cp_obj_loss)
    if not res.success:
        raise ValueError(res.message)
    obj_val = res.fun
    v = model1.get_voltages(res.x)
    s = model1.get_apparent_power_flows(res.x)
    pg = model1.get_p_gens(res.x)
    pq = model1.get_q_gens(res.x)
    reg = model1.get_regulator_taps()
    if area_name == 'area1':
        print(f"area1: ***Solver returned with v_down={v.iloc[-2:, 1:].to_numpy().astype(float)}, and s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    if area_name == "area2":
        print(f"area2: ***Solver returned with v_down={v.iloc[-1:, 1:].to_numpy().astype(float)}, and s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    if area_name == "area3":
        print(f"area3: ***Solver returned with s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    if area_name == "area4":
        print(f"area4: ***Solver returned with s_up={s.iloc[0, 2:].to_numpy().astype(complex)} in {res.runtime}s")
    return area_name, v, s, obj_val, pg, pq, reg

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
    pg_list = []
    qg_list = []
    reg_list = []

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


    for area in data.keys():
        anti_map[area] = {value: key for key, value in map[area].items()}
        # indexes = list(map[area].values())
        # indexes = list(set(data[area][3].index).intersection(set(indexes)))
        pg_df = data[area][3].copy()
        pg_df["id"] = pg_df.index
        pg_df.id = pg_df.id.apply(lambda x: anti_map[area][x])
        pg_list.append(pg_df)
    pg = pd.concat(pg_list)
    pg = pg.sort_values(by="id", ascending=True)
    pg.index = pg.id
    pg = pg.drop(columns=["id"])

    for area in data.keys():
        anti_map[area] = {value: key for key, value in map[area].items()}
        # indexes = list(map[area].values())
        # indexes = list(set(data[area][4].index).intersection(set(indexes)))
        qg_df = data[area][4].copy()
        qg_df["id"] = qg_df.index
        qg_df.id = qg_df.id.apply(lambda x: anti_map[area][x])
        qg_list.append(qg_df)
    qg = pd.concat(qg_list)
    qg = qg.sort_values(by="id", ascending=True)
    qg.index = qg.id
    qg = qg.drop(columns=["id"])

    for area in data.keys():
        anti_map[area] = {value: key for key, value in map[area].items()}
        # indexes = list(map[area].values())
        # indexes = list(set(data[area][4].index).intersection(set(indexes)))
        reg_df = data[area][5].copy()
        reg_df["id"] = reg_df.index
        reg_df.id = reg_df.id.apply(lambda x: anti_map[area][x])
        reg_list.append(reg_df)
    reg = pd.concat(reg_list)
    reg = reg.sort_values(by="id", ascending=True)
    reg.index = reg.id
    reg = reg.drop(columns=["id"])
    return v, s, pg, qg, reg

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

def admm_penalty_v_down(model: opf.LinDistModelModular, xk: cp.Variable, **kwargs):
    f = cp.Constant(0)
    down_nodes = model.bus_data.loc[model.bus_data.bus_type == opf.PQ_FREE, :].index
    for j in down_nodes:
        for ph in "abc":
            if not model.phase_exists(ph, j):
                continue
            v_down = model.bus_data.at[j, f"v_{ph}"]
            f += (xk[model.idx("v", j, ph)] - v_down**2)**2
    return f


def admm_penalty_p_down(model: opf.LinDistModelModular, xk: cp.Variable, **kwargs):
    f = cp.Constant(0)
    down_nodes = model.bus_data.loc[model.bus_data.bus_type == opf.PQ_FREE, :].index
    for j in down_nodes:
        for ph in "abc":
            if not model.phase_exists(ph, j):
                continue
            p_down = model.bus_data.at[j, f"pl_{ph}"]
            f += (xk[model.idx("pij", j, ph)] - p_down)**2
    return f


def admm_penalty_q_down(model: opf.LinDistModelModular, xk: cp.Variable, **kwargs):
    f = cp.Constant(0)
    down_nodes = model.bus_data.loc[model.bus_data.bus_type == opf.PQ_FREE, :].index
    for j in down_nodes:
        for ph in "abc":
            if not model.phase_exists(ph, j):
                continue
            p_down = model.bus_data.at[j, f"ql_{ph}"]
            f += (xk[model.idx("qij", j, ph)] - p_down)**2
    return f


def admm_penalty_v_up(model: opf.LinDistModelModular, xk: cp.Variable, **kwargs):
    f = cp.Constant(0)
    up_nodes = model.bus_data.loc[model.bus_data.bus_type == opf.SWING_FREE, :].index
    for j in up_nodes:
        for ph in "abc":
            if not model.phase_exists(ph, j):
                continue
            v_up = model.bus_data.at[j, f"v_{ph}"]
            f += (xk[model.idx("v", j, ph)] - v_up**2)**2
    return f

# def admm_penalty_p_up(model: opf.LinDistModelModular, xk: cp.Variable, **kwargs):
#     f = cp.Constant(0)
#     up_nodes = model.bus_data.loc[model.bus_data.bus_type == opf.SWING_FREE, :].index

def cp_obj_loss_relaxed(model: opf.LinDistModelModular, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    rho_up = kwargs.get("rho_up", kwargs.get("rho", 10000))
    rho_down = kwargs.get("rho_down", kwargs.get("rho", 0))
    return (opf.cp_obj_loss(model, xk, **kwargs)
            + 1/2*1e6*admm_penalty_v_up(model, xk, **kwargs)
            # + 1/2*1e6*admm_penalty_v_down(model, xk, **kwargs)
            + 1/2*1e6*admm_penalty_p_down(model, xk, **kwargs)
            + 1/2*1e6*admm_penalty_q_down(model, xk, **kwargs)
            )


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
        s12_from_2 = data['area2'][1].loc[data['area2'][1]['fb'] == "up", ["a", "b", "c"]].to_numpy()
        s13_from_3 = data['area3'][1].loc[data['area3'][1]['fb'] == "up", ["a", "b", "c"]].to_numpy()
        s24_from_4 = data['area4'][1].loc[data['area4'][1]['fb'] == "up", ["a", "b", "c"]].to_numpy()

        s12_from_1 = data['area1'][1].loc[data['area1'][1]['tb'] == "down1", ["a", "b", "c"]].to_numpy()
        s13_from_1 = data['area1'][1].loc[data['area1'][1]['tb'] == "down2", ["a", "b", "c"]].to_numpy()
        s24_from_2 = data['area2'][1].loc[data['area2'][1]['tb'] == "down", ["a", "b", "c"]].to_numpy()

        # s12_target = (s12_from_2 + s12_from_1)/2
        # s13_target = (s13_from_3 + s13_from_1)/2
        # s24_target = (s24_from_2 + s24_from_4)/2
        s12_target = s12_from_2
        s13_target = s13_from_3
        s24_target = s24_from_4

        # Update the shared variables with the latest results
        # area2_load_p.append(new_area2_load.real.copy())
        # area3_load_p.append(new_area3_load.real.copy())
        # area4_load_p.append(new_area4_load.real.copy())
        # area2_load_q.append(new_area2_load.imag.copy())
        # area3_load_q.append(new_area3_load.imag.copy())
        # area4_load_q.append(new_area4_load.imag.copy())
        area2_load_p.append(s12_target.real)
        area3_load_p.append(s13_target.real)
        area4_load_p.append(s24_target.real)
        area2_load_q.append(s12_target.imag)
        area3_load_q.append(s13_target.imag)
        area4_load_q.append(s24_target.imag)

        # Voltage updates
        v12_from_1 = data["area1"][0].iloc[-2, -3:].to_numpy()
        v12_from_2 = data["area2"][0].iloc[0, -3:].to_numpy()
        v13_from_1 = data["area1"][0].iloc[-1, -3:].to_numpy()
        v13_from_3 = data["area3"][0].iloc[0, -3:].to_numpy()
        v24_from_2 = data["area2"][0].iloc[-1, -3:].to_numpy()
        v24_from_4 = data["area4"][0].iloc[0, -3:].to_numpy()

        # v12_target = (v12_from_1 + v12_from_2)/2
        # v13_target = (v13_from_1 + v13_from_3)/2
        # v24_target = (v24_from_2 + v24_from_4)/2
        v12_target = v12_from_1
        v13_target = v13_from_1
        v24_target = v24_from_2

        area2_volt_v = data["area1"][0].iloc[-2, -3:].to_numpy()
        area3_volt_v = data["area1"][0].iloc[-1, -3:].to_numpy()
        area4_volt_v = data["area2"][0].iloc[-1, -3:].to_numpy()
        # area2_volt_v = np.vstack([data['area1'][0][key].loc[data['area1'][0][key].index == 38, ["a", "b", "c"]].to_numpy() for key in data['area1'][0].keys()])
        # area3_volt_v = np.vstack([data['area1'][0][key].loc[data['area1'][0][key].index == 39, ["a", "b", "c"]].to_numpy() for key in data['area1'][0].keys()])
        # area4_volt_v = np.vstack([data['area2'][0][key].loc[data['area2'][0][key].index == 19, ["a", "b", "c"]].to_numpy() for key in data['area2'][0].keys()])

        # area2_volt.append(area2_volt_v.copy())
        # area3_volt.append(area3_volt_v.copy())
        # area4_volt.append(area4_volt_v.copy())
        area2_volt.append(v12_target.copy())
        area3_volt.append(v13_target.copy())
        area4_volt.append(v24_target.copy())

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
            v_max_diff = max(
                np.max(abs(area2_volt[-1] - area2_volt[-2])),
                np.max(abs(area3_volt[-1] - area3_volt[-2])),
                np.max(abs(area4_volt[-1] - area4_volt[-2]))
            )
            v_boundary_error = max(
                np.max(abs(v12_from_1 - v12_from_2)),
                np.max(abs(v13_from_1 - v13_from_3)),
                np.max(abs(v24_from_2 - v24_from_4))
            )
            print(f"max_v_diff={v_max_diff}")
            print(f"max_diff={max_diff}")
            print(f"max boundary error = {v_boundary_error}")
            # print(area4_load_q[-3:])

            if max_diff <= 1e-3:
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
    v_dopf, s_dopf, pg_dopf, qg_dopf, reg_dopf = to_global(data, map)
    v_dopf.to_csv("v_dopf.csv")
    s_dopf.to_csv("s_dopf.csv")
    pg_dopf.to_csv("pg_dopf.csv")
    qg_dopf.to_csv("qg_dopf.csv")
    reg_dopf.to_csv("reg_dopf.csv")

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
    case.bus_data.cvr_p = 0.6
    case.bus_data.cvr_q = 1.5
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    case.gen_data.sa_max = 0.048
    case.gen_data.sb_max = 0.048
    case.gen_data.sc_max = 0.048
    # model = opf.LinDistModelPQ(
    model = opf.LinDistModelCapacitorRegulatorMI(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data
    )
    res = model.solve(opf.cp_obj_loss)
    # res = opf.cvxpy_solve(model, opf.cp_obj_loss)
    print(f"Time taken for COPF: {perf_counter() - tic1} seconds")
    print(f"objective value from COPF: {res.fun}")
    reg_result = model.get_regulator_taps()
    v = model.get_voltages(res.x)
    s = model.get_apparent_power_flows(res.x)
    pg = model.get_p_gens(res.x)
    qg = model.get_q_gens(res.x)
    reg_result.to_csv("reg_result.csv")
    v.to_csv("v_copf.csv")
    s.to_csv("s_copf.csv")
    pg.to_csv("pg_copf.csv")
    qg.to_csv("qg_copf.csv")
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



"""
using cvrp = 0.6 and cvrq=1.5

/home/gray/.virtualenvs/helics/bin/python /home/gray/git/CPMACosim/gridlabd/30_ders_advanced/dopf_reg2.py 
0
area1: ***Solver returned with v_down=[[0.97365537 0.99716623 0.98292704]
 [0.96864551 0.9985986  0.98139476]], and s_up=[0.81229191+0.44286172j 0.40087984+0.24554752j 0.55933471+0.30936831j] in 2.0010505139980523s
area2: ***Solver returned with v_down=[[0.98317536 1.01890884 1.00083748]], and s_up=[0.44590807+0.18964377j 0.3077691 +0.10417409j 0.30435582+0.08601362j] in 1.366202193999925s
area3: ***Solver returned with s_up=[0.15961998+0.00091195j 0.15889922+0.00270967j 0.08074183+0.00048949j] in 5.179980115000944s
area4: ***Solver returned with s_up=[0.34975598+0.03936696j 0.20963495+0.01879236j 0.24404736+0.02807686j] in 10.292585819999658s
1
area1: ***Solver returned with v_down=[[0.98501051 1.00254574 0.98980301]
 [0.98385272 1.00401872 0.98765689]], and s_up=[0.79616149+0.07865994j 0.38965189-0.00222355j 0.54404825+0.02564784j] in 1.9410434679994069s
area2: ***Solver returned with v_down=[[0.9664216  0.99424322 0.97505573]], and s_up=[0.42287481+0.01642521j 0.26031904+0.00129154j 0.29064348+0.00477239j] in 5.900231273000827s
area3: ***Solver returned with s_up=[0.15578253+0.00091195j 0.15444385+0.0019662j  0.07781983+0.00048949j] in 5.782311929000571s
area4: ***Solver returned with s_up=[0.35013102+0.03944182j 0.20962386+0.0188536j  0.24480561+0.02807346j] in 9.854833761997725s
max_v_diff=0.025781750295500006
max_diff=0.1732185628049309
max boundary error = 0.025781750295500006
2
area1: ***Solver returned with v_down=[[0.98889397 1.00329743 0.98951014]
 [0.986057   1.00446955 0.98708184]], and s_up=[0.77112911+0.0157896j  0.33784123-0.00240376j 0.52739608+0.010778j  ] in 1.8742959030023485s
area2: ***Solver returned with v_down=[[0.97776829 0.99965895 0.98190277]], and s_up=[0.42691138+0.01707767j 0.26154385+0.00129226j 0.293068  +0.00477225j] in 5.868376275000628s
area3: ***Solver returned with s_up=[0.15828245+0.00091195j 0.15533903+0.00241029j 0.07855814+0.00048949j] in 5.429990837998048s
area4: ***Solver returned with s_up=[0.35067207+0.03981195j 0.21004468+0.01878569j 0.24469196+0.02812641j] in 14.347852409999177s
max_v_diff=0.011346697787375626
max_diff=0.011346697787375626
max boundary error = 0.011346697787375626
3
area1: ***Solver returned with v_down=[[0.98879345 1.00334993 0.98942592]
 [0.98591254 1.00453624 0.98698003]], and s_up=[0.77755615+0.01584195j 0.33998295-0.00239965j 0.53050465+0.01077799j] in 2.1574589840020053s
area2: ***Solver returned with v_down=[[0.98161589 1.00044473 0.98160377]], and s_up=[0.42870163+0.01711606j 0.26214413+0.00129146j 0.29288247+0.00477453j] in 5.900735221999639s
area3: ***Solver returned with s_up=[0.15864789+0.00091195j 0.15541429+0.00243188j 0.07848991+0.00048949j] in 4.6706509979994735s
area4: ***Solver returned with s_up=[0.35038978+0.03973591j 0.20979281+0.01893199j 0.24485537+0.02815694j] in 10.4399028609987s
max_v_diff=0.0038475967916336007
max_diff=0.0038475967916336007
max boundary error = 0.0038475967916336007
4
area1: ***Solver returned with v_down=[[0.98873298 1.00338333 0.98942358]
 [0.98584617 1.00457324 0.98697595]], and s_up=[0.77966374+0.01584503j 0.34067001-0.00239946j 0.53025148+0.01077814j] in 2.002998493000632s
area2: ***Solver returned with v_down=[[0.98153149 1.00049301 0.98151269]], and s_up=[0.42839422+0.01710817j 0.26190331+0.00129318j 0.29302172+0.00477584j] in 6.3728429910006525s
area3: ***Solver returned with s_up=[0.15862386+0.00091195j 0.15542525+0.00243505j 0.07847793+0.00048949j] in 6.94870978200197s
area4: ***Solver returned with s_up=[0.34963458+0.0392542j  0.20997385+0.01885198j 0.24477753+0.02836269j] in 9.349801683998521s
max_v_diff=9.108471906837945e-05
max_diff=0.0007552043018843047
max boundary error = 9.108471906837945e-05
Converged after 4 iterations
total objective value for DOPF:0.034383657124708714
Time taken for DOPF: 132.70666570100002 seconds
/home/gray/git/CPMACosim/gridlabd/30_ders_advanced/dopf_reg2.py:288: FutureWarning: The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.
  reg = pd.concat(reg_list)
/home/gray/git/CPMACosim/gridlabd/30_ders_advanced/dopf_reg2.py:288: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  reg = pd.concat(reg_list)
Time taken for COPF: 43.72763145999852 seconds
objective value from COPF: 0.03480234772278558
a         0.0
b         0.0
c    0.001944
dtype: object
a    0.006424
b    0.000000
c    0.004484
dtype: float64
a    0.039900
b    0.026267
c    0.016499
dtype: float64

Process finished with exit code 0

"""