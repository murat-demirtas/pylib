#! usr/bin/python
import numpy as np
from scipy.spatial.distance import pdist, cdist
import scipy
from scipy.stats import ttest_ind, spearmanr, pearsonr
import statsmodels.api as sm
from transform import isd
#from mne.stats import permutation_t_test, fdr_correction

###############################################################################
############ Advanced tests
###############################################################################
'''
Dependent correlations
'''
def scorr(x, y, prc = 75):

    N = len(x)
    data = np.vstack((x, y))
    distance = np.sqrt(((data - np.tile(np.median(data,1), (N, 1)).T) ** 2).sum(0))

    distance_median = np.median(distance)
    iqr = np.percentile(distance, prc) - np.percentile(distance, 100 - prc)
    # lo = distance_median - 1.5*iqr
    hi = distance_median + 1.5 * iqr
    mask = np.where(distance < hi)
    r_skipped = pearsonr(x[mask], y[mask])

    #r_skipped = pearsonr(x, y)
    return r_skipped

def dep_corr(aa, bb, cc, Nb = 1000, type = 'spearman', alpha = 0.05, verbose = False):
    N = len(aa)
    if type == 'spearman':
        corr1 = spearmanr(aa, bb)
        corr2 = spearmanr(aa, cc)
    else:
        corr1 = scorr(aa, bb)
        corr2 = scorr(aa, cc)

    bootcorr1 = np.zeros(Nb)
    bootcorr2 = np.zeros(Nb)
    for b in range(Nb):
        index = np.random.randint(N, size=N)
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

    if verbose:
        print 'corr(a,b) = ' + str(corr1) + ' [' + str(boot1ci[0]) + ',' + str(boot1ci[1]) + ']'
        print 'corr(a,c) = ' + str(corr2) + ' [' + str(boot2ci[0]) + ',' + str(boot2ci[1]) + ']'
        print 'difference = ' + str(corr1[0] - corr2[0]) + ' [' + str(diffci[0]) + ',' + str(diffci[1]) + ']'
        print 'pvalue = ' + str(pval)

    result = {}
    result['corr_b'] = corr1
    result['corr_c'] = corr2
    result['diff'] = corr1[0] - corr2[0]
    result['ci_corr_b'] = boot1ci
    result['ci_corr_c'] = boot2ci
    result['ci_diff'] = diffci
    result['pvalue'] = pval

    return result


###############################################################################
############ General Linear Model
###############################################################################
'''
Linear Model
'''
def lm(y, x):
    # With numeric input
    model = sm.OLS(y, sm.add_constant(x, prepend=False), missing='drop')
    results = model.fit()
    beta = results.params
    pval = results.pvalues
    tval = results.tvalues
    return beta, pval, tval

'''
Regress out
'''
def regress_out(y, x):
    # With numeric input
    model = sm.OLS(y, sm.add_constant(x, prepend=False), missing='drop')
    results = model.fit()
    return results.resid

###############################################################################
############ Comparisons
###############################################################################
'''
Permutation t-test
'''
def permutation_ttest2(x, y, n_perm = 5000):
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
    return t_orig, (abs(t_perm)>abs(t_orig)).mean()

'''
FDR correction
'''
def fdr(pvals, alpha, method):
    return fdr_correction(pvals = pvals, alpha = alpha, method = method)

###############################################################################
############ DISTANCE METRICS (Correlations)
###############################################################################
def partial_corr(x, control):
    #import pdb; pdb.set_trace()
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


def fisher_z(r, N = None):
    # Perform fisher z-transform
    if N is None:
        return np.arctanh(r)
    else:
        return np.arctanh(r) / np.sqrt(N - 3)


def vcorrcoef(X, y):
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r


def paired_corr_mat(x, y, pvals=False):
    N = x.shape[0]
    if pvals:
        return np.array([pearsonr(x[range(0, ii) + range(ii + 1, N), ii],
                                     y[range(0, ii) + range(ii + 1, N), ii]) for ii in range(N)])
    else:
        return np.array([pearsonr(x[range(0, ii) + range(ii + 1, N), ii],
                                     y[range(0, ii) + range(ii + 1, N), ii])[0] for ii in range(N)])


def paired_corr(x, y, corr='spearman', pvals=False):
    N = x.shape[1]
    if corr == 'spearman':
        if pvals:
            return np.array([spearmanr(x[:, kk], y[:, kk]) for kk in range(N)])
        else:
            return np.array([spearmanr(x[:, kk], y[:, kk])[0] for kk in range(N)])
    else:
        if pvals:
            return np.array([pearsonr(x[:, kk], y[:, kk]) for kk in range(N)])
        else:
            return np.array([pearsonr(x[:, kk], y[:, kk])[0] for kk in range(N)])


def corrcoef(x, y, pvals=False):
    r = 0.0
    if x.ndim > 1:
        if x.shape[0] == x.shape[1]:
            idx = isd(x.shape[0])
            r = pearsonr(x[idx], y[idx])
    else:
        r = pearsonr(x, y)

    if pvals:
        return r
    else:
        return r[0]


def cosine_sim(x, y=None, axis=0):
    # Compute cosine similarity
    if axis == 1:
        x = x.transpose()
        if y is not None: y = y.transpose()

    if y is None:
        return 1. - pdist(x, 'cosine')
    else:
        assert (x.shape == y.shape)
        return 1. - cdist(x, y, 'cosine')


def xcov(x, y=None, lag=1):
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


def cov_to_corr(cov):
    # Covariance to correlation coefficient
    cov_ii = np.diag(cov)
    norm2 = np.outer(cov_ii, cov_ii)
    try:
        corr = cov / np.sqrt(norm2)
    except FloatingPointError:
        if np.min(norm2) < 0:
            msg = 'Trying to take sqrt of negative number!'
            raise ValueError(msg)
    return corr


def autocorr(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    return r / (variance * np.arange(n, 0, -1))


###############################################################################
############ Descriptive
###############################################################################
def demean(x):
    return x - np.transpose(np.tile(np.mean(x,axis=1),[x.shape[1],1]))

def standardize(x):
    mu = np.transpose(np.tile(np.mean(x, axis=1), [x.shape[1], 1]))
    sigma = np.transpose(np.tile(np.std(x, axis=1), [x.shape[1], 1]))
    return (x - mu)/sigma




