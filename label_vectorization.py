#!/usr/bin/env python
import numpy as np
from scipy.stats import entropy
import math
from sklearn.manifold import TSNE
import pdb

# Helper functions to read data, to get answer and probability vectors
def get_ans_pct_vectors(answer_counters):

    answer_list = []
    answer_pcts = []

    for k, v in answer_counters.items():
        answer_list.append(v)
        answer_pcts.append(answers2pct(v))

    ans_vectors = np.asarray(answer_list, dtype=int)
    pct_vectors = np.asarray(answer_pcts, dtype=float)

    #print(len(answer_list), ans_vectors.shape, len(answer_pcts), pct_vectors.shape)

    return pct_vectors

def answers2pct(answers):
    try:
        s = sum(answers)
        return [float(x)/s for x in answers]
    except ZeroDivisionError as e:
        return answers

# Helper functions to read data, to get answer vectors
def get_ans_vectors(tweetid_answer_counters):

    itemid_list, answer_list = zip(*tweetid_answer_counters.items())
    ans_vectors = np.asarray(answer_list, dtype=int)
    # print(len(itemid_list), len(answer_list), ans_vectors.shape)

    return ans_vectors

def tests(true_pct_vectors, prediction_proba_vectors):
    # Do similar things as
    # https://github.com/kobe2452/subjective_active_learning/blob/master/MultinomialCluster.py#L128
    #print(true_pct_vectors.shape, prediction_proba_vectors.shape)

    # xentropy = [entropy(x) for x in true_pct_vectors]
    xentropy = []
    for x in true_pct_vectors:
        if sum(x) == 0:
            xentropy.append(0)
        else:
            xentropy.append(entropy(x))
    tentropy = [entropy(t) for t in prediction_proba_vectors]
    xmax = [max(x) for x in true_pct_vectors]
    tmax = [max(t) for t in prediction_proba_vectors]

    if (len(xentropy) != len(xmax)) or (len(tentropy) != len(tmax)):
        print(len(xentropy), len(xmax), len(tentropy), len(tmax))

    return zip(xentropy, tentropy, xmax, tmax)

def get_assignments(test_true_vectors, test_pred_vectors):
    dist_by_cluster = [[0.0] * len(test_true_vectors[0]) for i in test_pred_vectors[0]]
    assignments_per_cluster = [0.0] * len(test_pred_vectors[0])
    cluster_assignments = [np.argmax(tpv) for tpv in test_pred_vectors]
    for i in range(len(test_true_vectors)):
        dist_by_cluster[cluster_assignments[i]] = [j + k for j,k in zip(dist_by_cluster[cluster_assignments[i]], test_true_vectors[i])]
        assignments_per_cluster[cluster_assignments[i]] += 1

    return cluster_assignments, dist_by_cluster, assignments_per_cluster

def get_perplexity(test_vectors, cluster_assignments, dist_by_cluster, assignments_per_cluster):
    dist_by_cluster = np.asarray([answers2pct(x) for x in dist_by_cluster])
    if np.sum(test_vectors, axis=1).shape == (len(test_vectors),):
        test_vectors = test_vectors
    else:
        test_vectors = np.asarray([answers2pct(x) for x in test_vectors])
    cluster_assignments = np.asarray(cluster_assignments)
    #print(test_vectors.shape, cluster_assignments.shape, dist_by_cluster.shape)

    epsilon =  0.00001
    tot = 0.0 #K-L Divergence
    for tv, ci in zip(test_vectors, cluster_assignments):
        for x, y in zip(tv, dist_by_cluster[ci]):
            x = x + epsilon
            y = y + epsilon
            tot += x * math.log(x/y)

    #cross_entropy = [sum([-x * math.log(y) ]) ]
    return (tot / len(test_vectors))

def use_tSNE_reduce_dimensions(vectors, target_dim):

    # http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne = TSNE(n_components=target_dim, init='pca', random_state=5, learning_rate=100.0)
    return tsne.fit_transform(vectors)
