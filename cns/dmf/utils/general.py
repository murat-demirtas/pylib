#! usr/bin/python

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import gaussian, fftconvolve
from scipy.optimize import curve_fit
from scipy.spatial.distance import cosine


def cosine_sim(corr1, corr2):
    assert(corr1.shape == corr2.shape)
    iu = np.triu_indices(corr1.shape[0], k=1)
    return 1. - cosine(np.arctanh(corr1[iu]), np.arctanh(corr2[iu]))


def xcorr(x, y, max_lag):
    """Computes cross-correlation between x and y up
    to max_lag, returning array of length 2 * max_lag + 1
    """

    # initialize output array, lags
    output = np.empty((2 * max_lag) + 1)
    lags = np.arange(-max_lag, max_lag + 1)

    # loop over lag values
    for ii, lag in enumerate(lags):

        # use slice indexing to get lagged time series
        if lag < 0:
            xi = x[:lag]
            yi = y[-lag:]
        elif lag > 0:
            xi = x[lag:]
            yi = y[:-lag]
        else:
            xi = x
            yi = y

        # correlate lagged arrays
        c = np.corrcoef(xi, yi)

        # add to output array
        output[ii] = c[0, 1]

    return output


def clean_builtins(my_dict):
    cleaned_dict = dict()
    for key, val in my_dict.items():
        if key[:2] != "__":
            cleaned_dict[key] = val
    return cleaned_dict


def prefix_keys(my_dict, prefix, sep="_"):
    prefixed_dict = dict()
    for key, val in my_dict.items():
        prefixed_dict[prefix + sep + key] = val
    return prefixed_dict


def autocorr(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    return r / (variance * np.arange(n, 0, -1))


def mask_diag(matr):
    mask = np.logical_not(np.identity(matr.shape[0])).astype(int)
    return matr * mask


def symmetrize(matr):
    assert(type(matr) == np.ndarray)
    return 0.5 * (matr + matr.T)


def pearson_sim(model_FC, emp_FC, use_diag=False):
    """Returns similarity of model correlation matrix to empirical
       FC (using Pearson's correlation coefficient). """

    # Want upper triangular part of symmetric matrix
    k = [1, 0][use_diag]
    iu = np.triu_indices(emp_FC.shape[0], k=k)

    return pearsonr(model_FC[iu], emp_FC[iu])[0]


def lagged_autocorr(sim_dict, max_lag, t_cutoff=10, BOLD=False):

    # time series, time resolution, number of cortical areas
    ts = sim_dict['S_E'] if not BOLD else sim_dict['y']
    tres = sim_dict['dt']
    nc = ts.shape[0]

    # convert lag in seconds to samples
    max_lag = int(max_lag / tres)

    # don't want data before steady-state reached
    n_cutoff = int(t_cutoff / tres)
    ts = ts[:, n_cutoff:]

    # initialize output
    acorr_matrix = np.empty((nc, max_lag+1))

    # cross-correlate
    for i in range(nc):
        acorr_matrix[i, :] = xcorr(ts[i, :], ts[i, :], max_lag=max_lag)

    # exponential decaying lambda function
    exp_decay = lambda t, tau: np.exp(-t/tau)

    time_lags = np.arange(max_lag + 1)*tres
    tscales = np.empty(nc)

    for i in range(nc):
        node_tscale = curve_fit(exp_decay, time_lags,
                                acorr_matrix[i, :], p0=[0.2], maxfev=1000)[0]
        tscales[i] = node_tscale

    return acorr_matrix, tscales


class Bundle(object):

    def __init__(self, attrDict):

        for key, attr in attrDict.items():
            if key[:2] != "__":
                # print 'Adding %s to Bundle' % key
                self.__setattr__(key, attr)

    def update(self, attrDict):

        for key, attr in attrDict.items():
            if key[:2] != "__":
                # print 'Adding %s to Bundle' % key
                self.__setattr__(key, attr)


def low_pass_filter(x, dt_sim=1e-4, n_save=50, sigma=0.1, window=1):
    """Uses a Gaussian smoothing kernel to perform low-pass filtering of
    time series x. The actual number of timesteps has to be scaled by n_save
    since not every simulated data point was saved due to memory constraints.
    """

    nt = len(x)
    # total_t = nt * dt_sim * n_save

    # construct normalized Gaussian smoothing kernel
    # sigma = 100-ms, for 0.1-ms timestep
    g = gaussian(int(window / (dt_sim * n_save)), int(sigma / (dt_sim * n_save)))
    g /= np.sum(g)

    return fftconvolve(x, g)[int(window / (dt_sim * n_save)) /
                             2:-int(window / (dt_sim * n_save)) / 2 + 1]


def cov_to_corr(cov, full_matrix=True):
    """ Generate correlation matrix from covariance matrix.
        full_matrix: set True if cov is the full 2n x 2n covariance
        matrix and you wish to return the covariance matrix of
        the upper left quadrant (i.e. the E-E block) """

    if full_matrix:
        nc = cov.shape[0] / 2
        cov_EE = cov[:nc, :nc]
    else:
        cov_EE = cov
    cov_ii = np.diag(cov_EE)
    norm2 = np.outer(cov_ii, cov_ii)
    try:
        corr = cov_EE / np.sqrt(norm2)
    except FloatingPointError:
        if np.min(norm2) < 0:
            msg = 'Trying to take sqrt of negative number! Check stability of system.'
            raise ValueError(msg)
    return corr


def fisher_z(r):
    return np.arctanh(r)


def weighted_pearson(corr_matrices, FC_emp, weights=1., use_diag=False):
    """Computes the Pearson correlation coefficient between
    the empirical FC and the model FC matrices in
    corr_matrices, of shape (nG, nc, nc). If a numpy array of
    weights is passed (e.g., cortical areas), they are normalized
    and used to weight each component.

    :rtype : np.ndarray """

    use_builtin = True  # No weights, use scipy.stats.pearsonr

    if isinstance(weights, np.ndarray):
        # print "Weighting by cortical areas..."
        assert (len(weights) == corr_matrices.shape[1])
        use_builtin = False

    # Number of trial G values
    nG = corr_matrices.shape[0]
    nc = corr_matrices.shape[1]

    # One scalar for each trial G value
    FC_sim_vals = np.empty(nG)

    # Only use upper triangular part of symmetric matrices
    k = [1, 0][use_diag]
    iu = np.triu_indices(nc, k=k)
    FC_emp_triu = FC_emp[iu]

    # Compute normalized area weights
    if not use_builtin:
        weights_ij = np.outer(weights, weights)
        weights_triu = weights_ij[iu]
        weights_triu /= np.sum(weights_triu)

    for i in xrange(nG):

        # Model prediction
        FC_model = corr_matrices[i, :, :]
        FC_model_triu = FC_model[iu]

        if use_builtin:
            p_coeff = pearsonr(FC_model_triu, FC_emp_triu)[0]
            FC_sim_vals[i] = p_coeff

        else:

            # Model deviation from mean
            model_weighted_mean = np.sum(FC_model_triu * weights_triu)
            FC_model_dev = weights_triu * (FC_model_triu - model_weighted_mean)

            # Empirical deviation from mean
            FC_emp_dev = FC_emp_triu - np.mean(FC_emp_triu)

            # Compute normalizations
            numerator = np.sum(FC_model_dev * FC_emp_dev)
            norm_emp = np.sqrt(np.sum(FC_emp_dev ** 2))
            norm_model = np.sqrt(np.sum(FC_model_dev ** 2))

            pcoeff = numerator / (norm_emp * norm_model)

            FC_sim_vals[i] = pcoeff

    return FC_sim_vals


def clean_mask(mask, n_pad=0):
    """ Cleans the output binary mask produced from the
    G - w parameter sweep (fills in holes). """

    mask_cleaned = np.copy(mask)

    nw, ng = mask.shape

    # column-wise submask
    for g_i in range(ng):

        try:
            first_unstable_w_ind = np.min(np.where(mask[:, g_i])[0])
        except ValueError:
            # no unstable inds at this g value
            continue

        # force all entries beyond this point to be nan
        mask_cleaned[max(0, first_unstable_w_ind - n_pad):, g_i:] = 1

    # row-wise submask
    for w_i in range(nw):

        try:
            first_unstable_g_ind = np.min(np.where(mask[w_i, :])[0])
        except ValueError:
            # no unstable inds at this w value
            continue

        # force all entries beyond this point to be nan
        mask_cleaned[w_i, max(0, first_unstable_g_ind - n_pad):] = 1

    return mask_cleaned
