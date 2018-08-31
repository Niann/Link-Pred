import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle


threshold = 10

train = {} # key: src   value: [dest1, dest2, ...]
vector_count = {} # indegree count

print ("reading train set")

with open("train.txt") as trainfile:
    for i, line in tqdm(enumerate(trainfile)):
        line_list = [int(k) for k in line[:-1].split("\t")]
        a = line_list[0]
        train[a] = []
        for b in line_list[1:]:
            train[a].append(b)
            vector_count[b] = vector_count.get(b,0)+1
        train[a] = list(set(train[a]))


print ("--------complete")
print ("generating dictionary")


# generate new node set
# filter by indegree threshold
new_set = set()
for i in vector_count:
    if vector_count[i]>threshold:
        new_set.add(i)

# add all source node
for i in train:
    new_set.add(i)


with open("raw.pickle","rb") as f:
    test = pickle.load(f)

for i,j,k,_ in test:
    new_set.add(j)
    new_set.add(k)

id2v = list(new_set) # [v1, v2, ...]
v2id = {} # key: vertex    value: index
for i,j in enumerate(id2v):
    v2id[j] = i

print ("length of new set:")
print (len(new_set))

# generate new node id dictionary
new_train = {} # key: index    value: set of connected nodes after filtering
for i in train:
    new_train[v2id[i]] = set([v2id[j] for j in train[i] if j in new_set])



new_test = {} # key: training sample id    value: [id for v_i, id for v_j]
for i,j,k,_ in test:
    new_test[i] = [v2id[j],v2id[k]]
    # remove true edge
    if v2id[k] in new_train[v2id[j]]:
        new_train[v2id[j]].remove(v2id[k])


tA = new_train.copy()
tB = {}
for i in new_train:
    if i not in tA[i]:
        tA[i].add(i)
    for j in new_train[i]:
        tB[j] = tB.get(j,set([]))
        tB[j].add(i)

print ("now processing...")


def sim(pair,tA,tB,l):
    vi, vj = pair
    tempA = np.zeros(l);
    tempB = np.zeros(l);
    tempA[list(tA[vi])] = 1/len(tA[vi]);
    if vj in tB:
        for i in tB[vj]:
            tempB[list(tA[i])] += 1/len(tB[vj])/len(tA[i]);
    return cosine_similarity([tempA, tempB])[0][1]
    #return tempA,tempB


l=len(new_set)
res = {}
for i in tqdm(new_test, ascii=True):
    vi,vj = new_test[i]
    res[i] = []
#    res[i].append(sim([vi,vj],tA,tB,l))
    res[i].append(sim([vj,vi],tB,tA,l))
    

with open("rec_sim.pickle","wb") as f:
    pickle.dump(res, f)






#
