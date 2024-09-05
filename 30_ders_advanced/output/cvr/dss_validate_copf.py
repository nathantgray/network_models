import distopf as opf
import pandas as pd
import numpy as np
def update_gens(parser, p_df, q_df):
    gen_flag = parser.dss.Generators.First()
    while gen_flag:
        name_phase = parser.dss.Generators.Name()
        phase = name_phase[-1]
        name = name_phase[:-1]
        p = p_df.loc[p_df.name.astype(str) == name, phase].to_numpy()[0]
        q = q_df.loc[q_df.name.astype(str) == name, phase].to_numpy()[0]
        parser.dss.Generators.kW(np.round(p*1e3, 3))
        parser.dss.Generators.kvar(np.round(q*1e3, 3))
        gen_flag = parser.dss.Generators.Next()


branch_data = pd.read_csv("branch_data.csv")
bus_data = pd.read_csv("bus_data.csv")
pg_copf = pd.read_csv("pg_copf.csv")
qg_copf = pd.read_csv("qg_copf.csv")
v_copf = pd.read_csv("v_copf.csv")
s_copf = pd.read_csv("s_copf.csv")
parser = opf.DSSParser("../../opendss/master.dss")
update_gens(parser, pg_copf, qg_copf)
parser.update()
v_copf.index += 1
v_diff = v_copf.copy()
v_diff.loc[:, ["a", "b", "c"]] = v_copf.loc[:, ["a", "b", "c"]] - parser.v_solved.loc[:, ["a", "b", "c"]]
print("v diff")
print(v_diff.loc[:, ["a", "b", "c"]].abs().max())
pl_copf = bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]]
ql_copf = bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]]
pl_dss = parser.bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]]
ql_dss = parser.bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]]
print("dopf total p loads")
print(pl_copf.sum())
print("dopf total q loads")
print(ql_copf.sum())
print("dss total p loads")
print(pl_dss.sum())
print("dss total q loads")
print(ql_dss.sum())
pg_dss = parser.gen_data.loc[:, ["pa", "pb", "pc"]]
p_substation = parser.s_solved.loc[parser.s_solved.tb=="150r", ["a", "b", "c"]]

p_loss = p_substation.to_numpy().flatten().real - pl_dss.sum().to_numpy() + pg_dss.sum().to_numpy()
print(sum(p_loss))
parser.v_solved.to_csv("v_dss_c.csv")
parser.s_solved.to_csv("s_dss_c.csv")
parser.branch_data.to_csv("branch_data_dss_c.csv", index=False)
parser.bus_data.to_csv("bus_data_dss_c.csv", index=False)
parser.gen_data.to_csv("gen_data_dss_c.csv", index=False)
parser.cap_data.to_csv("cap_data_dss_c.csv", index=False)
parser.reg_data.to_csv("reg_data_dss_c.csv", index=False)

# for i in range(branch_data.shape[0]):
#     print(f"dss: {parser.branch_data.fb[i]}->{parser.branch_data.tb[i]}  original: {branch_data.fb[i]}->{branch_data.tb[i]}")
# pass
