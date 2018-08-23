import numpy as np
import pickle

from feature_extractor import *
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

with open("matrix.pickle","rb") as f:
    matrix = pickle.load(f)

with open("raw.pickle","rb") as f:
    data = pickle.load(f)

SPLIT_RATIO = 0.9

def feature_generator(datum):

    if len(datum) != 2:
        print(len(datum))
    a, b = datum

    # node feature
    a_in = indegree(a, matrix)
    a_out = outdegree(a, matrix)
    b_in = indegree(b, matrix)
    b_out = indegree(b, matrix)

    # neighbouring feature
    neighbour = common_neighbour(a, b, matrix)
    jac = jaccard(neighbour, a, b, matrix)
    p_a = pref_attach(a, b, matrix)
    cos = cosine_sim(neighbour, p_a)
    adar = adamic_adar(a, b, matrix)

        # path feature
        #sim_r = sim_rank(a, b, matrix, 0)

        #X.append([a_in,a_out,b_in,b_out,neighbour,jac,p_a,cos,adar])

    return [a_in,a_out,b_in,b_out,neighbour,jac,p_a,cos,adar]

if __name__ ==  '__main__':

    print("start")
    pool = Pool(processes=4)
    train_test = pool.map(feature_generator, [(d[0],d[1]) for d in data])
    pool.close()
    print("end")

    labels = [d[2] for d in data]

    X, y = np.array(train_test), np.array(labels)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT_RATIO)
    X_train.dump("Xtrain")
    X_test.dump("Xtest")
    y_train.dump("ytrain")
    y_test.dump("ytest")
