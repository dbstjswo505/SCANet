import json
import math



with open("train_data_comp.json", 'r') as fget:
    data = json.load(fget)

for i in range(len(data)):
    sce = data[i][-1]
    if sce > 10:
        nsce = 1.0
    else:
        nsce = round(sce*(0.1)-0.1,2)
    data[i][-1] = nsce

with open("./out/train_data_comp.json", 'w') as fput:
    json.dump(data, fput)

with open("val_data_comp.json", 'r') as fget:
    data = json.load(fget)

for i in range(len(data)):
    sce = data[i][-1]
    if sce > 10:
        nsce = 1.0
    else:
        nsce = round(sce*(0.1)-0.1,2)
    data[i][-1] = nsce

with open("./out/val_data_comp.json", 'w') as fput:
    json.dump(data, fput)

