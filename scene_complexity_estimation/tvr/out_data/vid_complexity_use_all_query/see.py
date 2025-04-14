import json
import pdb
import torch

with open("./train_data_comp.json", 'r') as fget:
    data = json.load(fget)
M = 0
m = 0
cnt = torch.zeros((30),dtype=torch.long)
div = torch.arange((30))
div[0] = 1
for i in range(len(data)):
    sce = data[i][-1]
    cnt[sce] = cnt[sce] + 1
    if sce > M:
        M = sce
    if sce < m:
        m = sce
cnt_udt = cnt/div
pdb.set_trace()

