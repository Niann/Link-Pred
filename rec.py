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

        

test = []

with open("test-public.txt") as testfile:
    for i, line in enumerate(testfile):
        if i == 0:
            continue;
        line_list = [int(k) for k in line[:-1].split("\t")];
        test.append([line_list[1],line_list[2]])

print ("--------complete")
print ("generating dictionary")
        
vector_count = {}

for i in train:
    for j in train[i]:
        vector_count[j] = vector_count.get(j,0)+1

new_set = set()
for i in vector_count:
    if vector_count[i]>threshold:
        new_set.add(i)

for i,j in test:
    new_set.add(j)

for i in train:
    new_set.add(i)

with open("raw.pickle","rb") as f:
    ttrain = pickle.load(f)
    
for i,j,k,_ in ttrain:
    new_set.add(j)
    new_set.add(k)
    
id2v = list(new_set)
v2id = {}
for i,j in enumerate(id2v):
    v2id[j] = i
    
print ("length of new set:")
print (len(new_set))

new_train = {}
for i in train:
    new_train[v2id[i]] = [v2id[j] for j in train[i] if j in new_set]
    
new_test = {}
for i,j,k,_ in ttrain:
    new_test[i] = [v2id[j],v2id[k]]
    
tA = new_train.copy()
tB = {}
for i in new_train:
    if i not in tA[i]:
        tA[i].append(i)
    for j in new_train[i]:
        tB[j] = tB.get(j,[])
        tB[j].append(i)

print ("now processing...")


def get_feature(i,tA,l,layer=1):
    wlist = [[i,1]]
    temp = np.zeros(l)
    for i in range(layer):
        tlist = []
        for ni,scale in wlist:
            if ni in tA:
                nscale = scale/len(tA[ni])
                temp[tA[ni]] +=nscale
                if i < (layer-1):
                    for nj in tA[ni]:
                        tlist.append([nj,nscale])
            else:
                temp[ni]+=scale
                tlist.append([ni,scale])
            #print (1)
        wlist = tlist
    return np.array(temp)

def sim(pair,tA,tB,l):
    vi,vj = pair
    tempA = np.zeros(l);
    tempB = np.zeros(l);
    tempA[tA[vi]] = 1/len(tA[vi]);
    for i in tB[vj]:
        tempB[tA[i]] += 1/len(tB[vj])/len(tA[i]);
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































    

    


