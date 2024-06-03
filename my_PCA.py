import pandas as pd
from numpy.linalg import linalg
from sklearn.base import BaseEstimator
from sklearn.decomposition._pca import PCA
from sklearn.utils import *
import numpy as np
from sklearn.preprocessing import StandardScaler


def PCA_skl(X, n_components=2, random_state=0):
    # base = BaseEstimator()
    X = BaseEstimator()._validate_data(X, accept_sparse=['csr'], ensure_min_samples=2, dtype=[np.float32, np.float64])
    random_state = check_random_state(random_state)
    pca = PCA(n_components=n_components, svd_solver='randomized',
              random_state=random_state)
    # print(pca.explained_variance_ratio_)
    X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
    u = pca.fit_transform(X)
    return X_embedded, u


def PCA_my(X, K):
    X = X.T
    X_std = StandardScaler().fit_transform(X.T).T

    n = X_std.shape[1]
    cov_mat = X_std.dot(X_std.T) / (n - 1)

    cov_matrix = np.cov(X_std)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vecs_K = eig_vecs[:, 0:K]

    matrix_W = eig_vecs_K.T
    PCA_X = matrix_W.dot(X)

    return PCA_X.T, matrix_W


if __name__ == '__main__':
    clients_grad = pd.read_csv('../logs/2022-07-08/16.36.30/param/PCA_clients_10x4.csv', header=None)
    clients_grad = np.array(clients_grad)

    A = np.array([[1, 0, 0, 0], [0, 0, 0, 4], [0, 3, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]])

    U, sigma, VT = linalg.svd(clients_grad)
    U_K = U[:, 0:10]

    W = U_K.T

    PCA_clients = W.dot(clients_grad)

    print(A)
