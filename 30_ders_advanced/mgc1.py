
import multiprocessing as mp
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

def initialize():

    case = opf.DistOPFCase(
        data_path=Path.cwd(),
        gen_mult=2,
        load_mult=1,
        v_swing=1.05,
        v_max=1.05,
        v_min=0.95,
    )
    case.gen_data.a_mode = "CONSTANT_PQ"
    case.gen_data.b_mode = "CONSTANT_PQ"
    case.gen_data.c_mode = "CONSTANT_PQ"
    # case.gen_data.sa_max = case.gen_data.sa_max / 1.2
    # case.gen_data.sb_max = case.gen_data.sb_max / 1.2
    # case.gen_data.sc_max = case.gen_data.sc_max / 1.2
    model1 = opf.LinDistModelPQ(
        case.branch_data,
        case.bus_data,
        case.gen_data,
        case.cap_data,
        case.reg_data
    )
    res = opf.cvxpy_solve(model1, opf.cp_obj_loss)
    # res = model1.solve(opf.cp_obj_curtail)
    obj_val = res.fun
    v = model1.get_voltages(res.x)
    s = model1.get_apparent_power_flows(res.x)
    print(f"copf objective: {obj_val}")
    s12 = s.loc[126, ["a", "b", "c"]].to_numpy().astype(complex).flatten()
    s13 = s.loc[125, ["a", "b", "c"]].to_numpy().astype(complex).flatten()
    s24 = s.loc[128, ["a", "b", "c"]].to_numpy().astype(complex).flatten()
    v12 = v.loc[126, ["a", "b", "c"]].to_numpy().astype(float).flatten()
    v13 = v.loc[125, ["a", "b", "c"]].to_numpy().astype(float).flatten()
    v24 = v.loc[128, ["a", "b", "c"]].to_numpy().astype(float).flatten()
    area2_load_p = s.loc[126, ["a", "b", "c"]]
    return s12, s13, s24, v12, v13, v24

if __name__ == '__main__':
    s12, s13, s24, area2_volt, area3_volt, area4_volt = initialize()
    area2_load_p = np.real(s12)  # [-0.19255064, -0.36659203, -0.33063738]
    area2_load_q = np.imag(s12)  # [0.41237562, 0.33956541, 0.31216232]
    area3_load_p = np.real(s13)  # [-0.07858945, -0.08507705, -0.15988413]
    area3_load_q = np.imag(s13)  # [0.1553687, 0.19040997, 0.12507865]
    area4_load_p = np.real(s24)  # [-0.03640842, -0.17870877, -0.1439719 ]
    area4_load_q = np.imag(s24)  # [0.28715018, 0.24834041, 0.22938035]
    # area2_load_p = np.array([-0.19255064, -0.36659203, -0.33063738])
    # area2_load_q = np.array([0.41237562, 0.33956541, 0.31216232])
    # area3_load_p = np.array([-0.07858945, -0.08507705, -0.15988413])
    # area3_load_q = np.array([0.1553687, 0.19040997, 0.12507865])
    # area4_load_p = np.array([-0.03640842, -0.17870877, -0.1439719])
    # area4_load_q = np.array([0.28715018, 0.24834041, 0.22938035])
    # area2_load_p, area3_load_p, area4_load_p = [np.zeros(3)], [np.zeros(3)], [np.zeros(3)]
    # area2_load_q, area3_load_q, area4_load_q = [np.zeros(3)], [np.zeros(3)], [np.zeros(3)]
    # area2_load_p = np.ones(3)*-0.2
    # area2_load_q = np.ones(3)*0.3
    # area3_load_p = np.ones(3)*-0.07
    # area3_load_q = np.ones(3)*0.12
    # area4_load_p = np.ones(3)*-0.15
    # area4_load_q = np.ones(3)*0.23
    # area2_volt = np.array([0.96448149, 0.97351394, 0.96628513])
    # area3_volt = np.array([0.95991123, 0.97403011, 0.96601825])
    # area4_volt = np.array([0.9524924,  0.97046076, 0.95689746])
    # area2_volt, area3_volt, area4_volt = [np.ones(3)*0.95], [np.ones(3)*0.95], [np.ones(3)*0.95]
    area_name = "area2"

    case = opf.DistOPFCase(
        data_path=Path.cwd()/area_name,
        gen_mult=2,
        load_mult=1,
        v_swing=1.05,
        v_max=1.05,
        v_min=0.95,
    )
    case.gen_data.a_mode = "CONTROL_PQ"
    case.gen_data.b_mode = "CONTROL_PQ"
    case.gen_data.c_mode = "CONTROL_PQ"
    # case.gen_data.sa_max = case.gen_data.sa_max / 1.2
    # case.gen_data.sb_max = case.gen_data.sb_max / 1.2
    # case.gen_data.sc_max = case.gen_data.sc_max / 1.2

    if area_name == 'area1':
        for ph in "abc":
            case.bus_data.loc[40, f"pl_{ph}"] = area2_load_p["abc".index(ph)]
            case.bus_data.loc[40, f"ql_{ph}"] = area2_load_q["abc".index(ph)]
            case.bus_data.loc[41, f"pl_{ph}"] = area3_load_p["abc".index(ph)]
            case.bus_data.loc[41, f"ql_{ph}"] = area3_load_q["abc".index(ph)]
    elif area_name == 'area2':
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area2_volt
        for ph in "abc":
            case.bus_data.loc[19, f"pl_{ph}"] = area4_load_p["abc".index(ph)]
            case.bus_data.loc[19, f"ql_{ph}"] = area4_load_q["abc".index(ph)]
    elif area_name == 'area3':
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area3_volt
    elif area_name == 'area4':
        case.bus_data.loc[0, ["v_a", "v_b", "v_c"]] = area4_volt
    # model1 = opf.LinDistModelCapacitorRegulatorMI(
    #     case.branch_data,
    #     case.bus_data,
    #     case.gen_data,
    #     case.cap_data,
    #     case.reg_data
    # )
    # res = model1.solve(opf.cp_obj_curtail)

    model1 = opf.LinDistModelPQ(
        case.branch_data,
        case.bus_data,
        case.gen_data,
        case.cap_data,
        case.reg_data
    )
    res = opf.cvxpy_solve(model1, opf.cp_obj_loss)
    obj_val = res.fun
    v = model1.get_voltages(res.x)
    s = model1.get_apparent_power_flows(res.x)
    print(obj_val)
    # print(v)
    # print(s)