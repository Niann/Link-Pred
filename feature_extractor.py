from math import log

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
    return neighbour / pref_a

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

def sim_rank(a, b, matrix, level, gamma = 0.95):
    if a == b:
        return 1
    if a not in matrix or b not in matrix or level >= 5:
        return 0
    s = 0
    for a_ in matrix[a]:
        for b_ in matrix[b]:
            if (a_ in matrix and b_ in matrix) or a_ == b_:
                s += gamma * sim_rank(a_, b_, matrix, level+1) / (len(matrix[a]) * len(matrix[b]))
    return s
