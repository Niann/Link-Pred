from math import log, sqrt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def indegree(node, matrix):
    cnt = 0
    for src in matrix:
        if node in matrix[src]:
            cnt += 1
    return cnt

def outdegree(node, matrix):
    if node in matrix:
        return len(matrix[node])
    else:
        return 0

def common_neighbour(a, b, matrix):
    if a not in matrix or b not in matrix:
        return 0
    return len(matrix[a].intersection(matrix[b]))

def jaccard(neighbour, a, b, matrix):
    if neighbour == 0:
        return 0
    return neighbour / (len(matrix[a]) + len(matrix[b]) - neighbour)

def cosine_sim(neighbour, pref_a):
    if neighbour == 0 or pref_a == 0:
        return 0
    return neighbour / sqrt(pref_a)

def LHN(neighbour, pref_a):
    if neighbour == 0 or pref_a == 0:
        return 0
    return neighbour / pref_a

def dice_idx(neighbour, a, b, matrix):
    if neighbour == 0:
        return 0
    return neighbour / (len(matrix[a]) + len(matrix[b]))

def pref_attach(a, b, matrix):
    if a not in matrix or b not in matrix:
        return 0
    return len(matrix[a]) * len(matrix[b])

def adamic_adar(a, b, matrix):
    if a not in matrix or b not in matrix:
        return 0
    neighbour = matrix[a].intersection(matrix[b])
    connections = [len(matrix[n]) if n in matrix else 0 for n in neighbour]
    score = sum([log(1/(n+1e-2)) for n in connections])
    return score

def resource_allocation(a, b, matrix):
    if a not in matrix or b not in matrix:
        return 0
    neighbour = matrix[a].intersection(matrix[b])
    connections = [len(matrix[n]) if n in matrix else 0 for n in neighbour]
    score = sum([1/(n+1e-2) for n in connections])
    return score

def hub_promoted(neighbour, a, b, matrix):
    if neighbour == 0:
        return 0
    return neighbour / min(len(matrix[a]), len(matrix[b]))

def hub_depressed(neighbour, a, b, matrix):
    if neighbour == 0:
        return 0
    return neighbour / max(len(matrix[a]), len(matrix[b]))

def reverse_link(a, b, matrix):
    if b not in matrix:
        return 0
    if a in matrix[b]:
        return 1
    else:
        return 0

def sim_rank(a, b, matrix, level, gamma = 0.95):
    if a == b:
        return 1
    if a not in matrix or b not in matrix or level >= 2:
        return 0
    s = 0
    for a_ in matrix[a]:
        for b_ in matrix[b]:
            if (a_ in matrix and b_ in matrix) or a_ == b_:
                s += gamma * sim_rank(a_, b_, matrix, level+1) / (len(matrix[a]) * len(matrix[b]))
    return s

def propflow(a, b, matrix, level=3):
    found = set([a])
    new_search = [a]
    s1 = {a:1}
    s2 = {a:1}
    for l in range(level):
        old_search = new_search.copy()
        new_search = []
        #print(len(old_search))
        while len(old_search) != 0:
            v = old_search.pop(0)
            if v in matrix:
                node_input = s2[v]
                sum_output = len(matrix[v])
                for u in matrix[v]:
                    if u in matrix or u == b:
                        if l == 1 or u != b:
                            s1[u] = s1.get(u,0) + node_input * (1/sum_output)
                        if l > 0 or u != b:
                            s2[u] = s2.get(u,0) + node_input * (1/sum_output)
                        if u not in found:
                            found.add(u)
                            new_search.append(u)
    return s1.get(b, 0), s2.get(b, 0)

def recommend_vector(a,b,rec):
    tA = rec["A"]
    tB = rec["B"]
    vi,vj = a,b
    tempA = np.zeros(l);
    tempB = np.zeros(l);
    tempA[tA[vi]] = 1/len(tA[vi]);
    for i in tB[vj]:
        tempB[tA[i]] += 1/len(tB[vj])/len(tA[i]);
    return cosine_similarity([tempA, tempB])[0][1]
    
    
    
    
    
    
    
    
