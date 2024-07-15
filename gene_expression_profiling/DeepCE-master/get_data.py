import pandas as pd
import numpy as np
import os

file_path = './DeepCE/data/signature_test.csv'

data = pd.read_csv(file_path)
print(data)

filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

sigid = list(data.sig_id)
#print(sigid)
sigid = [sigid[i] for i in range(len(sigid)) if filter['time'] in sigid[i]]
print(len(sigid))

pertid = list(data.pert_id)
pertid = [pertid[i] for i in range(len(pertid)) if pertid[i] not in filter['pert_id']]
print(len(pertid))

perttype = list(data.pert_type)
perttype = [perttype[i] for i in range(len(perttype)) if perttype[i] in filter['pert_type']]
print(len(perttype))

cellid = list(data.cell_id)
cellid = [cellid[i] for i in range(len(cellid)) if cellid[i] in filter['cell_id']]
print(len(cellid))

pertidose = list(data.pert_idose)
pertidose = [pertidose[i] for i in range(len(pertidose)) if pertidose[i] in filter['pert_idose']]
print(len(pertidose))

newdata = data.loc[data.sig_id.isin(sigid) & data.pert_id.isin(pertid) & data.pert_type.isin(perttype) & data.cell_id.isin(cellid) & data.pert_idose.isin(pertidose)]
print(newdata)