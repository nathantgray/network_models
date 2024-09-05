#%%
import pandas as pd
import numpy as np
from pathlib import Path
from cpmac.branchflow.importer import get_powerdata
#%%
power_data = get_powerdata("/home/nathangray/PycharmProjects/ReviseRejected/gridlabd/PNNL_R3_distributed/powerdata.csv")
power_data.PgA = power_data.Pa
power_data.PgB = power_data.Pb
power_data.PgC = power_data.Pc
power_data.to_csv(
    "/home/nathangray/PycharmProjects/ReviseRejected/gridlabd/PNNL_R3_distributed/powerdata.csv", index=False)

