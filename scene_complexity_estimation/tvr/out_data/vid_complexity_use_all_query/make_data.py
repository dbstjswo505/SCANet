import json
import pdb


with open('./original/val_data.json', 'r') as ofile:
    src = json.load(ofile)

with open('./vid_queries_set/vid_queries_val.json', 'r') as jfile:
    data = json.load(jfile)


for i in range(len(src)):
    vid = src[i][0]
    complexity = len(data[vid])
    src[i].append(complexity)

with open('val_data_comp.json', 'w') as go:
    json.dump(src, go)

pdb.set_trace()
z=1
