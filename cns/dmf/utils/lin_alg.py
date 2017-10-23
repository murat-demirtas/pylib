#!/usr/bin/python

import numpy as np
from matplotlib.mlab import PCA


def change_basis(basis_vecs, matrix):
    # if basis_vecs orthogonal, can replace inverse with .T
    return np.dot(np.linalg.inv(basis_vecs), np.dot(matrix, basis_vecs))


def SVD(matrix):
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, s, Vt


def eigendecomp(matrix):
    evals, evecs = np.linalg.eig(matrix)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    return evals, evecs


def pc_analysis(data):

    # Each row is an observation, each column a variable
    nrows, ncols = data.shape
    if ncols > nrows:
        data = data.T

    results = PCA(data)

    fracs = results.fracs    # proportion of variance of each pc
    comps = results.Wt       # weight vector for projecting point into PCA space
    projections = results.Y  # projected into PCA space

    return fracs, projections, comps.T


def pad_zeros(matrix):
    n = matrix.shape[0]
    return np.pad(matrix, pad_width=((0, n), (0, n)),
                  mode='constant', constant_values=0)
