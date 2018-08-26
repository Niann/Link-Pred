import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle


threshold = 10

train = {}
vector_set = set()
vector_count = {}

print ("reading train set")

with open("train.txt") as trainfile:
    for i, line in tqdm(enumerate(trainfile)):
        line_list = [int(k) for k in line[:-1].split("\t")]
        a = line_list[0]
        vector_set.add(a)
        train[a] = []
        for b in line_list[1:]:
            train[a].append(b)
            vector_set.add(b)
            vector_count[b] = vector_count.get(b,0)+1
        train[a] = list(set(train[a]))

        
print ("--------complete")
print ("generating dictionary")


#generate new node set
new_set = set()
for i in vector_count:
    if vector_count[i]>threshold:
        new_set.add(i)

for i in train:
    new_set.add(i)

with open("raw.pickle","rb") as f:
    test = pickle.load(f)
    
for i,j,k,_ in test:
    new_set.add(j)
    new_set.add(k)
    
id2v = list(new_set)
v2id = {}
for i,j in enumerate(id2v):
    v2id[j] = i
    
print ("length of new set:")
print (len(new_set))

#generate new node id dictionary
new_train = {}
for i in train:
    new_train[v2id[i]] = set([v2id[j] for j in train[i] if j in new_set])
    

    
new_test = {}
for i,j,k,_ in test:
    new_test[i] = [v2id[j],v2id[k]]
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
    vi,vj = pair
    tempA = np.zeros(l);
    tempB = np.zeros(l);
    tempA[list(tA[vi])] = 1/len(tA[vi]);
    for i in tB[vj]:
        tempB[list(tA[i])] += 1/len(tB[vj])/len(tA[i]);
    return cosine_similarity([tempA, tempB])[0][1]
    #return tempA,tempB


l=len(new_set)
res = {}
for i in tqdm(new_test):
    pair = new_test[i]
    res[i] = sim(pair,tA,tB,l)

with open("rec_sim.pickle","wb") as f:
    pickle.dump(res, f)






#































    

    


