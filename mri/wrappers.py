import numpy as np
from scipy.special import erf
from utils.transform import subdiag, squareform
from scipy.stats import gamma

"""
linearization function for myelin
"""
def linearize_map(x):
    return erf((x - np.mean(x)) / x.std() / np.sqrt(2))


"""
Structural Connectivity
"""
def normalize_sc(x):
    for ii in range(x.shape[0]): x[ii, ii] = 0.0
    x /= x.max()
    return x



def redistributed_sc(sc, hemi='LR'):
    if hemi == 'LR':
        sc_l = sc[0]
        sc_r = sc[1]
        N = len(subdiag(sc_l))
        rnd_l = gamma.rvs(2.0, size=N)
        rnd_r = gamma.rvs(2.0, size=N)
        sc_norm_l = np.zeros(N)
        sc_norm_r = np.zeros(N)
        sc_norm_l[subdiag(sc_l).argsort()] = np.sort(rnd_l)
        sc_norm_r[subdiag(sc_r).argsort()] = np.sort(rnd_r)
        sc_norm = [squareform(sc_norm_l / sc_norm_l.max()), squareform(sc_norm_r / sc_norm_r.max())]
    else:
        N = len(subdiag(sc))
        rnd = gamma.rvs(2.0, size=N)
        sc_norm = np.zeros(N)
        sc_norm[subdiag(sc).argsort()] = np.sort(rnd)
        sc_norm = squareform(sc_norm / sc_norm.max())
    return sc_norm


"""
Inter-individual variability
"""
def inter_individual_variability(fc, measure='rank', raw = True):
    N_regions = fc.shape[0]
    N_subjects = fc.shape[2]
    fc_flat = np.empty((N_regions, N_regions-1, N_subjects))
    for s in xrange(N_subjects):
        for n in xrange(N_regions):
            index = np.concatenate((np.arange(n), np.arange(n + 1, N_regions)))
            fc_flat[n, :, s] = fc[n, index, s]
    if measure =='rank':
        if N_subjects < 3:
            fmv2 = np.array([spearmanr(fc_flat[ii])[0] for ii in xrange(N_regions)])
        else:
            fmv2 = np.array([subdiag(spearmanr(fc_flat[ii])[0]) for ii in xrange(N_regions)])
    else:
        fmv2 = np.array([subdiag(np.corrcoef(fc_flat[ii].T)) for ii in xrange(N_regions)])

    if raw:
        return fmv2
    else:
        return 1.0 - fmv2.mean(1)

"""
Analytical approximation for GSR
"""
def analytical_gsr(cov_mat):
    N = cov_mat.shape[0]
    cov_sum = cov_mat.sum(1)
    cov_sum_matrix = np.tile(cov_sum, (N, 1))
    return cov_mat - (cov_sum_matrix * cov_sum_matrix.T) / cov_mat.sum()

"""
Matrix operations
"""
def separate_lr(x, indices_L, indices_R):
    if x.ndim == 3:
        xl = np.zeros((len(indices_L), len(indices_R), x.shape[2]))
        xr = np.zeros((len(indices_L), len(indices_R), x.shape[2]))
        for ii in range(x.shape[2]):
            xdummy = np.copy(x[:, :, ii])
            xl[:, :, ii] = np.copy(xdummy[indices_L, :][:, indices_L])
            xr[:, :, ii] = np.copy(xdummy[indices_R, :][:, indices_R])
    elif x.ndim == 2:
        if x.shape[0] == x.shape[1]:
            xl = x[indices_L, :][:, indices_L]
            xr = x[indices_R, :][:, indices_R]
        else:
            xl = x[indices_L, :]
            xr = x[indices_R, :]
    else:
        xl = x[indices_L]
        xr = x[indices_R]
    return [xl, xr]


def merge_lr(x_l, x_r, indices_L, indices_R):
    N = len(indices_L) + len(indices_R)
    if x_l.ndim == 1:
        x_out = np.zeros(N)
        x_out[indices_L] = x_l
        x_out[indices_R] = x_r
    else:
        x_out = np.empty((N, x_l.shape[1]))
        for ii in range(x_l.shape[1]):
            dummy = np.zeros(N)
            dummy[indices_L] = x_l[:, ii]
            dummy[indices_R] = x_r[:, ii]
            x_out[:, ii] = np.copy(dummy)
    return x_out


def concat_lr(x):
    return np.hstack((subdiag(x[0]), subdiag(x[1])))

def concat_lr_multi(mat, indices_L, indices_R):
    N = mat.shape[0] / 2
    S = mat.shape[2]
    Nc = N * (N - 1) / 2

    mat_l = np.empty((Nc, S))
    mat_r = np.empty((Nc, S))
    mat_lr = np.empty((Nc * 2, S))

    for ii in range(S):
        dummy = np.copy(mat[:, :, ii])
        dummy_l = dummy[indices_L, :][:, indices_L]
        dummy_r = dummy[indices_R, :][:, indices_R]

        mat_l[:,ii] = subdiag(dummy_l)
        mat_r[:, ii] = subdiag(dummy_r)
        mat_lr[:, ii] = np.hstack((subdiag(dummy_l), subdiag(dummy_r)))

    return mat_l, mat_r, mat_lr

def subdiag_multi(mat):
    N = mat.shape[0]
    S = mat.shape[2]
    Nc = N * (N - 1) / 2

    mat_diag = np.empty((Nc, S))
    for ii in range(S):
        mat_diag[:, ii] = subdiag(mat[:,:,ii])
    return mat_diag

"""
Input Output
"""
def hcp_subjects(file):
    subjects = file['subjects'].value
    N_subjects = len(subjects)

    if isinstance(subjects[0], str):
        return subjects
    else:
        return [str(subjects[ii]) for ii in range(N_subjects)]

def hcp_get(file, subject, keys):
    if len(keys) == 1:
        return file[subject][keys[0]].value
    else:
        return [file[subject][key].value for key in keys]

