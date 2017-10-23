#! usr/bin/python
import numpy as np
from matplotlib.mlab import PCA
from sklearn.decomposition import PCA


def SVD(matrix):
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, s, Vt

def pca(data, *args, **kwargs):
    pca = PCA(*args, **kwargs)
    projections = pca.fit_transform(data)
    fracs = pca.explained_variance_ratio_    # proportion of variance of each pc
    comps = pca.components_.T      # weight vector for projecting point into PCA space
    return fracs, projections, comps

def change_basis(basis_vecs, matrix):
    # if basis_vecs orthogonal, can replace inverse with .T
    return np.dot(np.linalg.inv(basis_vecs), np.dot(matrix, basis_vecs))


def eigendecomp(matrix):
    # Eigendecomposition
    evals, evecs = np.linalg.eig(matrix)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    return evals, evecs

