import pandas as pd
import distopf as opf

from cpmac.branchflow.importer import get_powerdata, get_branchdata

if __name__ == "__main__":
    branch_data = get_branchdata(
        "/home/nathangray/PycharmProjects/ReviseRejected/gridlabd/30_ders"
    )
    """
    fb,tb,raa,rab,rac,rbb,rbc,rcc,xaa,xab,xac,xbb,xbc,xcc,type,name,status,s_base,v_ln_base,z_base,phases
    """
    branch_data["type"] = "overhead_line"
    branch_data["name"] = branch_data.tb

    def find_phases(row):
        phase = ""
        if row.raa != 0 or row.xaa != 0:
            phase += "a"
        if row.rbb != 0 or row.xbb != 0:
            phase += "b"
        if row.rcc != 0 or row.xcc != 0:
            phase += "c"
        return phase

    branch_data["phases"] = branch_data.apply(find_phases, axis=1)
    branch_data.to_csv("branch_data.csv", index=False)

    power_data = get_powerdata(
        "/home/nathangray/PycharmProjects/ReviseRejected/gridlabd/30_ders"
    )
    bus_data = pd.DataFrame(
        columns=[
            "id",
            "pl_a",
            "ql_a",
            "pl_b",
            "ql_b",
            "pl_c",
            "ql_c",
            "name",
            "bus_type",
            "v_a",
            "v_b",
            "v_c",
            "v_ln_base",
            "s_base",
            "v_min",
            "v_max",
            "cvr_p",
            "cvr_q",
            "phases",
            "has_gen",
            "has_load",
            "has_cap",
            "latitude",
            "longitude",
        ]
    )
    bus_data.id = power_data.id.to_numpy()
    bus_data.pl_a = power_data.Pa
    bus_data.ql_a = power_data.Qa
    bus_data.pl_b = power_data.Pb
    bus_data.ql_b = power_data.Qb
    bus_data.pl_c = power_data.Pc
    bus_data.ql_c = power_data.Qc
    bus_data.name = power_data.id
    bus_data.bus_type = "PQ"
    bus_data.v_a = 1
    bus_data.v_b = 1
    bus_data.v_c = 1
    bus_data.v_ln_base = 2401.77712
    bus_data.s_base = 1e6
    bus_data.v_min = 0.95
    bus_data.v_max = 1.05
    bus_data.cvr_p = 0
    bus_data.cvr_q = 0
    # bus_data.phases = "abc"
    for i, row in bus_data.iterrows():
        bus_data.at[i, "phases"] = branch_data.loc[
            branch_data.tb == row.id, "phases"
        ].to_numpy()
    bus_data.at[0, "phases"] = "abc"
    bus_data.to_csv("bus_data.csv", index=False)
    # cap_data = pd.DataFrame(columns=["id", "name", "qa", "qb", "qc", "phases"])
    gen_data = pd.DataFrame(
        columns=[
            "id",
            "name",
            "pa",
            "pb",
            "pc",
            "qa",
            "qb",
            "qc",
            "sa_max",
            "sb_max",
            "sc_max",
            "phases",
            "qa_max",
            "qb_max",
            "qc_max",
            "qa_min",
            "qb_min",
            "qc_min",
            "a_mode",
            "b_mode",
            "c_mode",
        ]
    )
    power_data_gens = power_data.loc[
        power_data.loc[:, ["PgA", "PgB", "PgC"]].sum(axis=1) > 0, ["PgA", "PgB", "PgC"]
    ].copy()
    gen_data.loc[:, "id"] = power_data_gens.index.to_numpy() + 1
    gen_data.loc[:, ["pa", "pb", "pc"]] = power_data_gens.loc[
        :, ["PgA", "PgB", "PgC"]
    ].to_numpy()
    gen_data.qa = 0
    gen_data.qb = 0
    gen_data.qc = 0
    gen_data.sa_max = gen_data.pa * 1.2
    gen_data.sb_max = gen_data.pb * 1.2
    gen_data.sc_max = gen_data.pc * 1.2
    gen_data.qa_max = 100
    gen_data.qb_max = 100
    gen_data.qc_max = 100
    gen_data.qa_min = -100
    gen_data.qb_min = -100
    gen_data.qc_min = -100
    gen_data.phases = "abc"
    gen_data.a_mode = opf.CONTROL_PQ
    gen_data.b_mode = opf.CONTROL_PQ
    gen_data.c_mode = opf.CONTROL_PQ
    gen_data.to_csv("gen_data.csv", index=False)
