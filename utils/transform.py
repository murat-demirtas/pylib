import numpy as np
from scipy.spatial.distance import squareform


def isd(N):
    # Return upper or lower diagonal mask
    # of a square matrix with dimension N
    return np.triu_indices(N, k=1)

def subdiag(matrix):
    N = matrix.shape[0]
    return matrix[np.triu_indices(N, k=1)]

def symmetrize(matr):
    # Return symmetric version of the matrix
    assert(type(matr) == np.ndarray)
    return 0.5 * (matr + matr.T)

def sqform(matrix):
    return squareform(matrix)


def pad_zeros(matrix):
    # Matrix padding
    n = matrix.shape[0]
    return np.pad(matrix, pad_width=((0, n), (0, n)),
                  mode='constant', constant_values=0)

