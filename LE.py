
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D


def rbf(dist, t = 1.0):
    return np.exp(-(dist/t))

def cal_pairwise_dist(x):

    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)

    return dist

def cal_rbf_dist(data, n_neighbors = 2, t = 1):

    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1+n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W

def le(data, snn = 0, n_dims = 2, n_neighbors = 8, t = 1.0):
    '''

    :param data: (n_samples, n_features)
    :param n_dims: target dim
    :param n_neighbors: k nearest neighbors
    :param t: a param for rbf
    :return:
    '''
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)

    W = W * (snn + 1)

    D = np.zeros_like(W)
    for i in range(N):
        D[i,i] = np.sum(W[i])
    D_inv = np.linalg.pinv(D) 
    L = D - W

    eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))
    sort_index_ = np.argsort(eig_val)

    eig_val = eig_val[sort_index_]

    j = 0
    while eig_val[j] < 1e-6:
        j+=1

    sort_index_ = sort_index_[j:j+n_dims]
    eig_val_picked = eig_val[j:j+n_dims]
    eig_vec_picked = eig_vec[:, sort_index_]

    X_ndim = eig_vec_picked
    return X_ndim
