import json
import pdb

with open('./original/train_data.json', 'r') as jfile:
    data = json.load(jfile)
out = dict()
for i in range(len(data)):
    vid = data[i][0]
    query = data[i][-1]
    if vid not in out.keys():
        out[vid] = [query]
    else:
        out[vid].append(query)
with open('./vid_queries_set/vid_queries_train.json', 'w') as jout:
    json.dump(out, jout)
pdb.set_trace()
z=1
