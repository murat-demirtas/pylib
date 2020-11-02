#! usr/bin/python
import numpy as np
from scipy.spatial.distance import pdist, cdist
import scipy
from scipy.stats import ttest_ind, spearmanr, pearsonr
import statsmodels.api as sm
# statsmodels: Seabold, Skipper and Joser Perktold, 2010


def lm(y, x):
    """
    Perform linear regression
    :param y: data
    :param x: predictors
    :return: beta, pval, tval
    """
    model = sm.OLS(y, sm.add_constant(x, prepend=False), missing='drop')
    results = model.fit()
    beta = results.params
    pval = results.pvalues
    tval = results.tvalues
    return beta, pval, tval


def regress_out(y, x):
    """
    Regress out x from y
    :param y: data
    :param x: predictor
    :return: data after regressing out x
    """
    model = sm.OLS(y, sm.add_constant(x, prepend=False), missing='drop')
    results = model.fit()
    return results.resid


def scorr(x, y, prc = 75):
    """
    Skipped Pearson correlation between parameters x and y
    :param prc: percentile
    :return: skipped Pearson correlation
    """
    '''
    Skipped Pearson correlation
    '''

    N = len(x)
    data = np.vstack((x, y))
    distance = np.sqrt(((data - np.tile(np.median(data,1), (N, 1)).T) ** 2).sum(0))
    distance_median = np.median(distance)
    iqr = np.percentile(distance, prc) - np.percentile(distance, 100 - prc)
    hi = distance_median + 1.5 * iqr
    mask = np.where(distance < hi)
    r_skipped = pearsonr(x[mask], y[mask])
    return r_skipped


def dep_corr(aa, bb, cc, Nb = 1000, type = 'spearman', alpha = 0.05):
    """
    Dependent correlation
    :param aa:
    :param bb:
    :param cc:
    :param Nb:
    :param type:
    :param alpha:
    :param verbose:
    :return:
    """
    n = len(aa)
    if type == 'spearman':
        corr1 = spearmanr(aa, bb)
        corr2 = spearmanr(aa, cc)
    else:
        corr1 = scorr(aa, bb)
        corr2 = scorr(aa, cc)

    bootcorr1 = np.zeros(Nb)
    bootcorr2 = np.zeros(Nb)
    for b in range(Nb):
        index = np.random.randint(n, size=n)
        if type == 'spearman':
            bootcorr1[b] = spearmanr(aa[index], bb[index])[0]
            bootcorr2[b] = spearmanr(aa[index], cc[index])[0]
        else:
            bootcorr1[b] = scorr(aa[index], bb[index])[0]
            bootcorr2[b] = scorr(aa[index], cc[index])[0]

    hi = np.floor((1.0 - alpha / 2.0) * Nb + .5)
    lo = np.floor((alpha / 2.0) * Nb + .5)

    bootdiff = bootcorr1 - bootcorr2
    bootdiff_sorted = np.sort(bootdiff)
    diffci = [bootdiff_sorted[lo], bootdiff_sorted[hi]]

    bootcorr1_sorted = np.sort(bootcorr1)
    bootcorr2_sorted = np.sort(bootcorr2)

    boot1ci = [bootcorr1_sorted[lo], bootcorr1_sorted[hi]]
    boot2ci = [bootcorr2_sorted[lo], bootcorr2_sorted[hi]]

    pval_raw = (bootdiff < 0).mean()
    pval = 2.0 * np.min((pval_raw, 1.0 - pval_raw))

    result = {}
    result['corr_b'] = corr1
    result['corr_c'] = corr2
    result['diff'] = corr1[0] - corr2[0]
    result['ci_corr_b'] = boot1ci
    result['ci_corr_c'] = boot2ci
    result['ci_diff'] = diffci
    result['pvalue'] = pval

    return result


def permutation_ttest2(x, y, n_perm = 5000):
    """
    Performs 2 sample permutation t-test
    :param x: samples 1
    :param y: samples 2
    :param n_perm: number of permutations
    :return:
    """
    t_orig, p_orig = ttest_ind(x,y)
    zs = np.concatenate([x, y])
    lx,ly,lz = len(x), len(y), len(zs)
    x2 = np.empty((lx, n_perm))
    y2 = np.empty((ly, n_perm))
    for j in range(n_perm):
        perms = np.random.permutation(lz)
        x2[:,j] = zs[perms[:lx]]
        y2[:,j] = zs[perms[lx:]]
    t_perm, p_perm = ttest_ind(x2, y2)
    return t_orig, (abs(t_perm) > abs(t_orig)).mean()


def partial_corr(x, control):
    """
    Computes partial correlations
    :param x:
    :param control:
    :return:
    """
    N = x.shape[0]
    T = x.shape[1]
    r = np.zeros((N, T))
    for ii in range(N):
        model = sm.OLS(x[ii], sm.add_constant(control, prepend=False),missing='drop')
        r[ii,:] = model.fit().resid

    corr = np.zeros((len(x),len(x)))
    p = np.zeros((len(x), len(x)))
    for ii in range(len(x)):
        for jj in range(len(x)):
            dummy = scipy.stats.pearsonr(r[ii,:],r[jj,:])
            corr[ii,jj] = dummy[0]
            p[ii,jj] = dummy[1]
    return corr, p


def partial_corr2(y, x, control):
    """
    Computes partial correlations between x and y
    :param y:
    :param x:
    :param control:
    :return:
    """
    model = sm.OLS(y, sm.add_constant(control, prepend=False), missing='drop')
    r0 = model.fit().resid
    r = np.zeros(len(x))
    p = np.zeros(len(x))
    for ii in range(len(x)):
        model = sm.OLS(x[ii], sm.add_constant(control, prepend=False),missing='drop')
        r1 = model.fit().resid
        dummy = scipy.stats.pearsonr(r0, r1)
        r[ii] = dummy[0]
        p[ii] = dummy[1]
    return r, p


def cosine_sim(x, y=None, axis=0):
    """
    Compute cosine similarity
    :param x:
    :param y:
    :param axis:
    :return:
    """
    if axis == 1:
        x = x.transpose()
        if y is not None: y = y.transpose()

    if y is None:
        return 1. - pdist(x, 'cosine')
    else:
        assert (x.shape == y.shape)
        return 1. - cdist(x, y, 'cosine')


def xcov(x, y=None, lag=1):
    """
    Computes cross-covariance
    :param x:
    :param y:
    :param lag:
    :return:
    """
    # If x is matrix
    if np.ndim(x) > 1:
        y = x if y is None else y

        N = np.shape(x)[0]
        c = np.zeros((N,N))

        if lag < 0:
            xi = x[:,:lag]
            yi = y[:,-lag:]
        elif lag > 0:
            xi = x[:, lag:]
            yi = y[:, :-lag]
        else:
            xi = x
            yi = y

        for i in range(N):
            for j in range(N):
                x = np.cov(xi[i,:],yi[j,:])
                c[i][j] = x[0,1]
        return c

    # If x is vector
    if np.ndim(x) == 1:
        y = x if y is None else y
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
        c = np.cov(xi, yi)
        # add to output array
        return c[0, 1]


def xcorr(x, y=None, lag=1):
    """
    Computes cross-correlation
    Inputs are each column, outputs are each row.
    :param x:
    :param y:
    :param lag:
    :return:
    """

    # If x is matrix
    if np.ndim(x) > 1:
        y = x if y is None else y

        N = np.shape(x)[0]
        c = np.zeros((N,N))

        if lag < 0:
            xi = x[:,:lag]
            yi = y[:,-lag:]
        elif lag > 0:
            xi = x[:, lag:]
            yi = y[:, :-lag]
        else:
            xi = x
            yi = y

        for i in range(N):
            for j in range(N):
                x = np.corrcoef(xi[i,:],yi[j,:])
                c[i][j] = x[0,1]
        return c

    # If x is vector
    if np.ndim(x) == 1:
        y = x if y is None else y
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
        return c[0, 1]