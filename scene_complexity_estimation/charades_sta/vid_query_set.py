import json
import pdb

with open('./original/test.json', 'r') as jfile:
    data = json.load(jfile)

out = dict()
for i in range(len(data)):
    vid = data[i][0]
    query = data[i][-1]
    if vid not in out.keys():
        out[vid] = [query]
    else:
        out[vid].append(query)

with open('./vid_query_set/vid_query_test.json', 'w') as jout:
    json.dump(out, jout)

with open('./original/train.json', 'r') as jfile:
    data = json.load(jfile)

out = dict()
for i in range(len(data)):
    vid = data[i][0]
    query = data[i][-1]
    if vid not in out.keys():
        out[vid] = [query]
    else:
        out[vid].append(query)

with open('./vid_query_set/vid_query_train.json', 'w') as jout:
    json.dump(out, jout)
