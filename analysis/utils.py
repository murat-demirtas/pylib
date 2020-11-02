import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.special import erf
from scipy.signal import hilbert
from math import pi


def demean(ts):
    """
    Subtract mean from the data
    :param ts: data matrix (NxT)
    :return: demeaned data
    """
    return ts - np.transpose(np.tile(np.mean(ts, axis=1), [ts.shape[1], 1]))


def autocorr(ts):
    """
    Computes autocorrelation of ts
    :param ts:
    :return:
    """
    n = len(ts)
    variance = ts.var()
    ts = ts - ts.mean()
    r = np.correlate(ts, ts, mode='full')[-n:]
    return r / (variance * np.arange(n, 0, -1))


def scrub(ts, index=None):
    """
    Scrubs timeseries
    :param ts: data matrix (NxT)
    :param index: time indices to scrub
    :return: scrubbed data
    """
    for ii in index:
        ts[:,ii] = (ts[:, ii-1] + ts[:, ii+1])/2
    return ts


def hilbert_transform(ts, crop=0):
    """
    Performs Hilbert transform to data
    :param ts:  data matrix (NxT)
    :param crop: time indices from begin and end to crop
    :return: phase, amplitude and Kuramoto order parameter
    """
    n = ts.shape[0]
    t = ts.shape[1]
    tcropped = np.arange(crop, t - crop)
    h = hilbert(ts, axis=1)[:, tcropped]
    phase = np.angle(h)
    amplitude = np.abs(h)
    kop = np.abs(np.sum((np.cos(phase) + 1j * np.sin(phase)) / n, axis=0))
    return phase, amplitude, kop


def phase_lock(ts, crop=0, method='difference'):
    """
    Computes phase lock values based on hilbert transform
    method defines the normalization of phase differences
    :param ts: data matrix (NxT)
    :param crop: time indices from begin and end to crop
    :param method: string: difference or cosine
    :return plv: phase-locking value
    """
    n = ts.shape[0]
    t = ts.shape[1]
    isd = np.triu_indices(n, k=1)
    n_diag = int(n * (n - 1) / 2)

    phase, amplitide, _ = hilbert_transform(ts, crop=crop)

    delta_phase = np.empty((n_diag, t))
    for ii in range(t):
        phase_matrix = np.tile(phase[:, ii], (n, 1))
        delta_phase[:, ii] = phase_matrix[isd] - phase_matrix.transpose()[isd]

    if method == 'difference':
        plv = abs(delta_phase)
        plv[np.where(plv > pi)] = 2 * pi - plv[np.where(plv > pi)]
        plv = 1 - plv / pi
    else:
        plv = np.cos(delta_phase)

    return plv


def sliding_window(ts, window=10, step=5):
    """
    Computes dynamic FC using sliding window analysis
    :param ts: data matrix (NxT)
    :param window: window size
    :param step: window step size
    :return dfc: dynamic fc
    """
    N = ts.shape[0]
    T = ts.shape[1]
    isd = np.triu_indices(N, k=1)

    slide = np.arange(0, T - window, step)
    dfc = np.zeros((N, len(slide)))
    for ii, t in enumerate(slide):
        dfc[:, ii] = np.corrcoef(ts[:, t:(t + window)])[isd]

    return dfc


def linearize_map(map):
    """
    Linearization function for myelin
    :param map: a gradient or map to linearize
    :return: linearized map
    """
    return erf((map - np.mean(map)) / map.std() / np.sqrt(2))


def normalize_psd(psd):
    """
    Normalizes power spectral density by its sum across signals
    :param psd: power spectral density
    :return: normalized psd
    """
    n = psd.shape[0]
    for ii in range(n):
        psd[ii, :] = psd[ii, :] / psd[ii, :].sum()
    return psd

def normalize_sc(sc):
    """
    Removes diagonal elements and normalizes structural connectivity
    :param: structural connectivity matrix
    :return: normalized structural connectivity
    """
    for ii in range(sc.shape[0]): sc[ii, ii] = 0.0
    sc /= sc.max()
    return sc


def subdiag(matrix, upper=True):
    """
    Return upper or lower diagonal
     of a square matrix with dimension N
    :param matrix: square matrix
    :param upper: returns upper diagonal if True
    :return: returns diagonal of a square matrix
    """
    N = matrix.shape[0]

    if upper:
        return matrix[np.triu_indices(N, k=1)]
    else:
        return matrix[np.tril_indices(N, k=-1)]


def subdiag_multi(matrix):
    """
    Return upper or lower diagonal
     of multiple square matrices with dimension N
    :param matrix: A set of square matrices (NxNxS)
    :return: Upper diagonal of each matrix (N(N-1)/2 x S)
    """
    N = matrix.shape[0]
    S = matrix.shape[2]
    Nc = N * (N - 1) / 2

    mat_diag = np.empty((Nc, S))
    for ii in range(S):
        mat_diag[:, ii] = subdiag(matrix[:,:,ii])

    return mat_diag


def symmetrize(matrix):
    """
    Symmetrize a square matrix
    :param matrix: square matrix
    :return: symmetric version of the matrix
    """
    assert(type(matrix) == np.ndarray)
    return 0.5 * (matrix + matrix.T)


def inter_individual_variability(fc, measure='rank', raw = True):
    """
    Computes inter-individual variability (Mueller et al., 2013)
    :param fc: functional connectivity (NxNxS), where S is number of subjects
    :param measure: Calculates Spearman correlation if 'rank', otherwise Pearson correlation
    :param raw: Returns 1 - average inter-individual variability if False
    :return: inter-individual variability
    """
    n = fc.shape[0]
    s = fc.shape[2]
    fc_flat = np.empty((n, n-1, s))

    for ii in range(s):
        for jj in range(n):
            index = np.concatenate((np.arange(jj), np.arange(jj + 1, n)))
            fc_flat[jj, :, ii] = fc[n, index, ii]

    if measure =='rank':
        if n < 3:
            fmv2 = np.array([spearmanr(fc_flat[ii])[0] for ii in range(n)])
        else:
            fmv2 = np.array([subdiag(spearmanr(fc_flat[ii])[0]) for ii in range(n)])
    else:
        fmv2 = np.array([subdiag(np.corrcoef(fc_flat[ii].T)) for ii in range(n)])

    if raw:
        return fmv2
    else:
        return 1.0 - fmv2.mean(1)


def analytical_gsr(cov):
    """
    Analytical approximation for global signal regression (GSR)
    :param cov: covariance matrix
    :return: covariance matrix after GSR
    """
    n = cov.shape[0]
    cov_sum = cov.sum(1)
    cov_sum_matrix = np.tile(cov_sum, (n, 1))
    return cov - (cov_sum_matrix * cov_sum_matrix.T) / cov.sum()


def vcorrcoef(x, y):
    """
    Correlation coefficient of X (X by S) to y
    :param x:
    :param y:
    :return:
    """
    xm = np.reshape(np.mean(x, axis=1), (x.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((x - xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((x - xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r

def pcorrcoef(x, y, corr='spearman', pvals=False):
    """
    Pair-to-pair correlation coefficient
    :param x:
    :param y:
    :param corr:
    :param pvals:
    :return:
    """
    n = x.shape[1]
    if corr == 'spearman':
        if pvals:
            return np.array([spearmanr(x[:, kk], y[:, kk]) for kk in range(n)])
        else:
            return np.array([spearmanr(x[:, kk], y[:, kk])[0] for kk in range(n)])
    else:
        if pvals:
            return np.array([pearsonr(x[:, kk], y[:, kk]) for kk in range(n)])
        else:
            return np.array([pearsonr(x[:, kk], y[:, kk])[0] for kk in range(n)])


def cov_to_corr(cov):
    """
    Generate correlation matrix from covariance matrix.
    :param cov: covariance matrix
    :return: correlation matrix
    """
    cov_ii = np.diag(cov)
    norm2 = np.outer(cov_ii, cov_ii)
    return cov / np.sqrt(norm2)

def fisher_z(r, N = None):
    """
    Computes Fisher z-transform
    :param r: correlation array or matrix
    :param N: number of timepoints
    :return: Fisher z-transformed correlations
    """
    if N is None:
        return np.arctanh(r)
    else:
        return np.arctanh(r) / np.sqrt(N - 3)