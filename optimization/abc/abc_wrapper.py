import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal
import sys
from tools.io import Data


def calc_weights(t_prev, t_curr, tau_2, w_old, prior="None"):
    """
    Calculates importance weights
    """
    weights_new = np.zeros_like(w_old)
    if len(t_curr.shape) == 1:
        norm = np.zeros_like(t_curr)
        for i, T in enumerate(t_curr):
            for j in xrange(t_prev[0].size):
                norm[j] = stats.norm.pdf(T, loc=t_prev[0][j],
                                     scale=tau_2)
            weights_new[i] = prior[0].pdf(T)/sum(w_old * norm)

        return weights_new/weights_new.sum()

    else:
        norm = np.zeros(t_prev.shape[1])
        for i in xrange(t_curr.shape[1]):
            prior_prob = np.zeros(t_curr[:, i].size)
            for j in xrange(t_curr[:, i].size):
                prior_prob[j] = prior[j].pdf(t_curr[:, i][j])
            #assumes independent priors
            p = prior_prob.prod()

            for j in xrange(t_prev.shape[1]):
                #import pdb; pdb.set_trace()
                norm[j] = stats.multivariate_normal.pdf(t_curr[:, i],
                                                        mean=t_prev[:, j],
                                                        cov=tau_2)

            weights_new[i] = p/sum(w_old * norm)

        return weights_new/weights_new.sum()

def weighted_covar(x, w):
    """
    Calculates weighted covariance matrix
    :param x: 1 or 2 dimensional array-like, values
    :param w: 1 dimensional array-like, weights
    :return C: Weighted covariance of x or weighted variance if x is 1d
    """
    sumw = w.sum()
    assert_almost_equal(sumw, 1.0)
    if len(x.shape) == 1:
        assert x.shape[0] == w.size
    else:
        assert x.shape[1] == w.size
    sum2 = np.sum(w**2)

    if len(x.shape) == 1:
        xbar = (w*x).sum()
        var = sum(w * (x - xbar)**2)
        return var * sumw/(sumw*sumw-sum2)
    else:
        xbar = [(w*x[i]).sum() for i in xrange(x.shape[0])]
        covar = np.zeros((x.shape[0], x.shape[0]))
        for k in xrange(x.shape[0]):
            for j in xrange(x.shape[0]):
                for i in xrange(x.shape[1]):
                    covar[j,k] += (x[j,i]-xbar[j])*(x[k,i]-xbar[k]) * w[i]

        return covar * sumw/(sumw*sumw-sum2)

def effective_sample_size(w):
    """
    Calculates effective sample size
    :param w: array-like importance sampleing weights
    :return: float, effective sample size
    """
    sumw = sum(w)
    sum2 = sum (w**2)
    return sumw*sumw/sum2


if __name__ == '__main__':
    arguments = sys.argv[1]
    arguments = arguments.split("_")

    m_type = arguments[0]
    hemi = arguments[1]
    if int(arguments[2]) == 1:
        session = 'session_1'
    elif int(arguments[2]) == 2:
        session = 'session_2'
    else:
        session = 'all'

    if int(arguments[3]) == 1:
        gradient = 'linearized'
    else:
        gradient = 'raw'

    measure = arguments[4] + '_' + arguments[5]

    gradient_direction = arguments[6]

    n_samples = int(arguments[7])
    n_outputs = int(arguments[8])
    iteration = int(arguments[9])

    data = Data()
    data.append_to_output('abc_' + m_type + '_' + hemi + '_' + session + '_' + measure + '_' + gradient + '_' + gradient_direction + '_' + str(n_samples))

    if iteration > 1:
        file_prev = data.load('iteration_'+str(iteration-1)+'.hdf5', from_path=data.output_dir)
        theta_prev = file_prev['theta'].value
        weights_prev = file_prev['weights'].value
        tau_squared_prev = file_prev['tau_squared'].value
        epsilon_prev = file_prev['epsilon'].value
        file_prev.close()

    p_theta = []
    ind_fit = []
    distance = np.empty(1)
    average_fc_fit = np.empty(1)
    n_accepted = np.empty(n_outputs)
    n_total = np.empty(n_outputs)
    for ii in range(n_outputs):
        results = data.load('samples_'+str(ii+1)+'.npy', from_path=data.output_dir)
        p_theta += [results[0]]
        distance = np.hstack((distance, results[1]))
        n_accepted[ii] = results[2]
        n_total[ii] = results[3]
        ind_fit += [results[4]]
        average_fc_fit = np.hstack((average_fc_fit, results[5]))

    theta = np.hstack([p_theta[p] for p in range(n_outputs)])
    individuals = np.vstack([ind_fit[p] for p in range(n_outputs)])
    n_accepted = n_accepted.sum()
    n_total = n_total.sum()
    distance = distance[1:]
    average_fc_fit = average_fc_fit[1:]
    maxfits = individuals.max(0)

    av_theta = theta.mean(1)
    """
    Report
    """
    print 'Iteration:' + str(iteration)
    print 'Max individual fit range: ' + str(maxfits.min()) + '-' + str(maxfits.max())
    print 'Mean individual fit: ' + str(individuals.mean()) + ', Max. individual fit: ' + str(maxfits.mean())
    print 'Av FC fit: ' + str(average_fc_fit.mean())
    print 'Acceptance rate: ' + str(n_accepted / n_total)
    print 'Theta:' + str(av_theta)

    if m_type == 'heterogeneous':
        if hemi == 'LR':
            prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5), stats.uniform(0.001, 5.0), stats.uniform(0.0, 15.0), stats.uniform(0.001, 5.0),
                     stats.uniform(0.001, 5.0)]
        else:
            prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5), stats.uniform(0.001, 5.0), stats.uniform(0.0, 15.0),
                     stats.uniform(0.001, 5.0)]
    else:
        if hemi == 'LR':
            prior = [stats.uniform(0.001, 5.), stats.uniform(0.001, 15.0), stats.uniform(0.001, 5.0), stats.uniform(0.001, 5.0)]
        else:
            prior = [stats.uniform(0.001, 5.), stats.uniform(0.001, 15.0), stats.uniform(0.001, 5.0)]


    if iteration==1:
        tau_squared = 2 * np.cov(theta)
        weights = np.ones(theta.shape[1]) * 1.0/theta.shape[1]
    else:
        weights = calc_weights(theta_prev, theta, tau_squared_prev, weights_prev, prior=prior)
        tau_squared = 2 * weighted_covar(theta, weights)

    #import pdb;

    #pdb.set_trace()
    epsilon_new = stats.scoreatpercentile(distance, per=75)
    if iteration > 1:
        if epsilon_new < epsilon_prev:
            epsilon = epsilon_new
        else:
            epsilon = epsilon_prev
    else:
        epsilon = epsilon_new

    effective_sample = effective_sample_size(weights)



    file_out = data.save('iteration_'+str(iteration)+'.hdf5')
    file_out.create_dataset('individual_fit', data=individuals)
    file_out.create_dataset('theta', data=theta)
    file_out.create_dataset('distance', data=distance)
    file_out.create_dataset('weights', data=weights)
    file_out.create_dataset('tau_squared', data=tau_squared)
    file_out.create_dataset('epsilon', data=epsilon)
    file_out.create_dataset('ess', data=effective_sample)
    file_out.create_dataset('n_accepted', data=n_accepted.sum())
    file_out.create_dataset('n_total', data=n_total.sum())
    file_out.create_dataset('av_fc_fit', data=average_fc_fit)
    file_out.close()

