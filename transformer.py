import numpy as np
import pickle

from feature_extractor import *
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

with open("matrix.pickle","rb") as f:
    matrix = pickle.load(f)

with open("raw.pickle","rb") as f:
    data = pickle.load(f)

with open("recommend.pickle","rb") as f:
    rec = pickle.load(f)

SPLIT_RATIO = 0.9

def feature_generator(datum):

    a, b, l = datum

    # node feature
    a_in = indegree(a, matrix)
    a_out = outdegree(a, matrix)
    b_in = indegree(b, matrix)
    b_out = indegree(b, matrix)

    # neighbouring feature
    neighbour = common_neighbour(a, b, matrix)
    jac = jaccard(neighbour, a, b, matrix)
    dice = dice_idx(neighbour, a, b, matrix)
    p_a = pref_attach(a, b, matrix)
    cos = cosine_sim(neighbour, p_a)
    lhn = LHN(neighbour, p_a)
    adar = adamic_adar(a, b, matrix)
    ra = resource_allocation(a, b, matrix)
    reverse = reverse_link(a, b, matrix)
    hp = hub_promoted(neighbour, a, b, matrix)
    hd = hub_depressed(neighbour, a, b, matrix)

    # path feature
    #sim_r = sim_rank(a, b, matrix, 0)

    flow2, flow3 = propflow(a, b, matrix)
    re = recommend_vector(a,b,rec)
    #print(flow)
    #return flow
    return [a_in,a_out,b_in,b_out,neighbour,jac,dice,p_a,cos,lhn,adar,reverse,hp,hd,flow2,flow3,l]

def logger(res):
    train_test.append(res)
    if len(train_test) % (len(data)//100) == 0:
        print("{:.2%} done".format(len(train_test)/len(data)))

if __name__ ==  '__main__':

    train_test = []
    print("start")
    pool = Pool(processes=4)
    #pool = Pool()
    for item in data:
        pool.apply_async(feature_generator, args=[item], callback=logger)
    pool.close()
    pool.join()
    print("end")

    X, y = np.array([d[:-1] for d in train_test]), np.array([d[-1] for d in train_test])
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT_RATIO)
    X_train.dump("Xtrain")
    X_test.dump("Xtest")
    y_train.dump("ytrain")
    y_test.dump("ytest")
