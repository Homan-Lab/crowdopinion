from scipy.spatial import distance
from scipy.stats import entropy
from keras import metrics
import functools
import numpy as np
import tensorflow as tf

# Distance measures: KLDivergence, Chebyshev, Clark, Canberra
# Similarity measures: cosine similarity, intersection similarity
# Assumes the probability distributions passed in are np arrays

epsilon = 0.00001

def chebyshev(y_pred, y_true):

    diff = np.abs(y_pred - y_true)
    mx = np.max(diff, 1)
    distance = np.mean(mx)

    return distance

def chebyshev_scipy(y_pred, y_true):

    n_samples = y_pred.shape[0]

    l = list()
    for i in range(n_samples):
        l.append(distance.chebyshev(y_true[i], y_pred[i]))

    return np.mean(l)

def clark(y_pred, y_true):

    y_pred += epsilon
    y_true += epsilon

    n = np.square(y_true - y_pred)
    d = np.square(y_true + y_pred)

    res = np.mean(np.sqrt(np.sum(n/d, 1)))

    y_pred -= epsilon
    y_true -= epsilon

    return res

def canberra(y_pred, y_true):

    y_pred += epsilon
    y_true += epsilon

    n = np.abs(y_true - y_pred)
    d = y_true + y_pred

    sum_vec = np.sum(n/d , 1)

    res = np.mean(sum_vec)

    y_pred -= epsilon
    y_true -= epsilon

    return res

def canberra_scipy(y_pred, y_true):

    n_samples = y_pred.shape[0]

    l = list()
    for i in range(n_samples):
        l.append(distance.canberra(y_true[i], y_pred[i]))

    return np.mean(l)

def kldivergence(y_pred, y_true):

    # TODO: normalize distributions if they dont sum to 1
    y_pred += epsilon
    y_true += epsilon

    inner_lg = np.log(y_true / y_pred)
    temp = y_true * inner_lg
    dist = np.sum(temp, 1)
    res = np.mean(dist)

    y_pred -= epsilon
    y_true -= epsilon

    return res

def kldivergence_scipy(y_pred, y_true):

    n_samples = y_pred.shape[0]

    l = list()
    for i in range(n_samples):
        l.append(entropy(y_true[i], y_pred[i]))

    return np.mean(l)

def cosine_similarity(y_pred, y_true):

    y_pred += epsilon
    y_true += epsilon

    inner = dot(y_true, y_pred)

    d1 = np.sqrt(dot(y_true, y_true))
    d2 = np.sqrt(dot(y_pred, y_pred))

    div = inner / (d1 * d2)

    res =  np.mean(div)

    y_pred -= epsilon
    y_true -= epsilon

    return 1 - res

def cosine_similarity_scipy(y_pred, y_true):

    n_samples = y_pred.shape[0]

    l = list()
    for i in range(n_samples):
        l.append(distance.cosine(y_true[i], y_pred[i]))

    return np.mean(l)

def dot(v1, v2):
    # computes dot product of two tensors(vectors)
    return np.sum(v1 * v2, 1)   # element wise


def intersection_similarity(y_pred, y_true):

    distance = np.sum(np.minimum(y_pred, y_true), 1)

    res =  np.mean(distance)

    return 1 - res

def get_all_metrics():
    metrics_all = [chebyshev_scipy, clark, canberra_scipy,
                    kldivergence_scipy, cosine_similarity_scipy, intersection_similarity]

    metrics_kld_cheby = [kldivergence_scipy, chebyshev]



    return metrics_kld_cheby

if __name__ == '__main__':

    # y_true = np.asarray([[0.2, 0.3, 0.5],
    #                     [0.2, 0.5, 0.3],
    #                     [0.9, 0.1, 0.0],
    #                     [0.7, 0.2, 0.1]])
    #
    # y_pred = np.asarray([[0.2, 0.3, 0.5],
    #                      [0.2, 0.4, 0.4],
    #                      [0.7, 0.2, 0.1],
    #                      [0.5, 0.3, 0.2]])

    y_true = np.asarray([[1, 0, 0], [0, 1.0, 0.0], [0, 0, 1]])
    y_pred = np.asarray([[1, 0, 0], [0, 0.9, 0.1], [0, 0, 1]])

    print(chebyshev(y_pred, y_true))
    print(chebyshev_scipy(y_pred, y_true))
    print(clark(y_true, y_pred))
    print(canberra(y_pred, y_true))
    print(canberra_scipy(y_pred, y_true))
    print(kldivergence(y_pred, y_true))
    print(kldivergence_scipy(y_pred, y_true))
    print(cosine_similarity(y_pred, y_true))
    print(cosine_similarity_scipy(y_pred, y_true))
    print(intersection_similarity(y_pred, y_true))
    # print(top3_acc(y_pred, y_true).eval())
