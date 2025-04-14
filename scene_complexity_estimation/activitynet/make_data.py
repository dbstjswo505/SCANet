import json
import pdb

with open('./original/test_data.json', 'r') as ofile:
    src = json.load(ofile)

with open('./vid_query_set/vid_query_test.json', 'r') as jfile:
    data = json.load(jfile)

for i in range(len(src)):
    vid = src[i][0]
    complexity = len(data[vid])
    src[i].append(complexity)

with open('./out_data/test_data_complexity.json', 'w') as go:
    json.dump(src, go)


with open('./original/train_data.json', 'r') as ofile:
    src = json.load(ofile)

with open('./vid_query_set/vid_query_train.json', 'r') as jfile:
    data = json.load(jfile)

for i in range(len(src)):
    vid = src[i][0]
    complexity = len(data[vid])
    src[i].append(complexity)

with open('./out_data/train_data_complexity.json', 'w') as go:
    json.dump(src, go)

with open('./original/val_data.json', 'r') as ofile:
    src = json.load(ofile)

with open('./vid_query_set/vid_query_val.json', 'r') as jfile:
    data = json.load(jfile)

for i in range(len(src)):
    vid = src[i][0]
    complexity = len(data[vid])
    src[i].append(complexity)

with open('./out_data/val_data_complexity.json', 'w') as go:
    json.dump(src, go)

