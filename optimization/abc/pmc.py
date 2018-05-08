import numpy as np
from utils.io import Data
from cns.dmf import model as dmf_model
from abc import ABCMeta, abstractmethod
from scipy import stats
from numpy.testing import assert_almost_equal
import os


class Pmc(object):
    __metaclass__ = ABCMeta

    def __init__(self, output_directory, appendices=False):
        self.data = Data()
        self.data.append_to_output(output_directory)
        if os.path.exists(self.data.output_dir + 'iteration_1.hdf5'):
            self.iteration = int(os.popen("ls " + self.data.output_dir + "iteration*" + " | wc -l").read()) + 1
        else:
            self.iteration = 1
        self.set_prior()
        self.appendices = appendices

    @abstractmethod
    def get_appendices(self, run_id):
        """
        Provide any other parameters to save
        :return: 
        """

    @abstractmethod
    def set_prior(self):
        """
        Provide prior
        self.prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5)]...etc.
        :return: 
        """

    @abstractmethod
    def run_particle(self, theta):
        """
        Run particle as:
        self.model.set('J_NMDA_EI',theta[0])
        self.model.set('J_NMDA_rec',theta[1])
        ...
        :param theta: 
        :return: 
        """

    @abstractmethod
    def generate_data(self):
        """
        Run particle as:
        self.model.moments_method(BOLD=True)
        self.model.get('corr_bold')
        ... 
        :return: 
        """

    @abstractmethod
    def distance_function(self, synthetic_data):
        """
        Run particle as:
        self.model.moments_method(BOLD=True)
        self.model.get('corr_bold')
        ... 
        :return: 
        """

    def set(self, sc, fc=None, gradient=None,
            n_particles=10, rejection_threshold=None,
            *args, **kwargs):
        self.model = dmf_model.Dmf(sc, gradient = gradient, *args, **kwargs)
        self.fc_objective = fc
        self.n_particles = n_particles
        self.rejection_threshold = rejection_threshold

    def draw_theta(self, theta_prev=None, weights=None, tau_squared=None):
        theta = None
        unstable = True
        if theta_prev is None:
            while unstable:
                theta = []
                for p in self.prior:
                    theta.append(p.rvs())

                if not (np.array(theta) < 0).any():
                    self.run_particle(theta)
                    unstable = self.model.check_stability()
        else:
            theta_star = theta_prev[:, np.random.choice(xrange(0, theta_prev.shape[1]),
                                                        replace=True, p=weights / weights.sum())]
            while unstable:
                theta = stats.multivariate_normal.rvs(theta_star, tau_squared)
                if not (np.array(theta) < 0).any():
                    self.run_particle(theta)
                    unstable = self.model.check_stability()
        return theta


    def run_sampler(self, epsilon, run_id, theta_prev=None, weights=None, tau_squared=None):
        posterior, distances = [], []
        trial_count, accepted_count = 0, 0
        while accepted_count < self.n_particles:
            theta = self.draw_theta(theta_prev=theta_prev, weights=weights, tau_squared=tau_squared)
            synthetic_data = self.generate_data()
            distance = self.distance_function(synthetic_data)
            trial_count += 1

            if distance < epsilon:
                if self.appendices:
                    self.get_appendices(run_id)

                accepted_count += 1
                posterior.append(theta)
                distances.append(distance)
            else:
                pass

        posterior = np.asarray(posterior).T
        distances = np.asarray(distances)
        return (posterior, distances,
                accepted_count, trial_count)


    def run(self, run_id):
        if self.iteration > 1:
            file_prev = self.data.load('iteration_' + str(self.iteration - 1) + '.hdf5',
                                       from_path=self.data.output_dir)
            theta_prev = file_prev['theta'].value
            weights_prev = file_prev['weights'].value
            tau_squared_prev = file_prev['tau_squared'].value
            epsilon = file_prev['epsilon'].value
            file_prev.close()

            results = self.run_sampler(epsilon, run_id, theta_prev=theta_prev, weights=weights_prev,
                                        tau_squared=tau_squared_prev)
        else:
            results = self.run_sampler(self.rejection_threshold, run_id)

        self.data.save('samples_' + str(run_id) + '.npy', results)


    def wrap(self, n_outputs):
        if self.iteration > 1:
            file_prev = self.data.load('iteration_' + str(self.iteration - 1) + '.hdf5',
                                       from_path=self.data.output_dir)
            theta_prev = file_prev['theta'].value
            weights_prev = file_prev['weights'].value
            tau_squared_prev = file_prev['tau_squared'].value
            epsilon_prev = file_prev['epsilon'].value
            file_prev.close()

        p_theta = []
        #ind_fit = []
        distance = np.empty(1)
        #average_fc_fit = np.empty(1)
        n_accepted = np.empty(n_outputs)
        n_total = np.empty(n_outputs)
        for ii in range(n_outputs):
            results = self.data.load('samples_' + str(ii + 1) + '.npy', from_path=self.data.output_dir)
            p_theta += [results[0]]
            distance = np.hstack((distance, results[1]))
            n_accepted[ii] = results[2]
            n_total[ii] = results[3]
            #ind_fit += [results[4]]
            #average_fc_fit = np.hstack((average_fc_fit, results[5]))

        theta = np.hstack([p_theta[p] for p in range(n_outputs)])
        #individuals = np.vstack([ind_fit[p] for p in range(n_outputs)])
        n_accepted = n_accepted.sum()
        n_total = n_total.sum()
        distance = distance[1:]
        #average_fc_fit = average_fc_fit[1:]
        #maxfits = individuals.max(0)

        print theta.mean(1)
        """
        Report
        
        print 'Iteration:' + str(iteration)
        print 'Max individual fit range: ' + str(maxfits.min()) + '-' + str(maxfits.max())
        print 'Mean individual fit: ' + str(individuals.mean()) + ', Max. individual fit: ' + str(maxfits.mean())
        print 'Av FC fit: ' + str(average_fc_fit.mean())
        print 'Acceptance rate: ' + str(n_accepted / n_total)
        print 'Theta:' + str(av_theta)
        """

        if self.iteration == 1:
            tau_squared = 2 * np.cov(theta)
            weights = np.ones(theta.shape[1]) * 1.0 / theta.shape[1]
        else:
            weights = self.calc_weights(theta_prev, theta, tau_squared_prev, weights_prev, prior=self.prior)
            tau_squared = 2 * self.weighted_covar(theta, weights)

        # import pdb;
        # pdb.set_trace()
        epsilon_new = stats.scoreatpercentile(distance, per=75)
        if self.iteration > 1:
            if epsilon_new < epsilon_prev:
                epsilon = epsilon_new
            else:
                epsilon = epsilon_prev
        else:
            epsilon = epsilon_new

        effective_sample = self.effective_sample_size(weights)

        file_out = self.data.save('iteration_' + str(self.iteration) + '.hdf5')
        file_out.create_dataset('theta', data=theta)
        file_out.create_dataset('distance', data=distance)
        file_out.create_dataset('weights', data=weights)
        file_out.create_dataset('tau_squared', data=tau_squared)
        file_out.create_dataset('epsilon', data=epsilon)
        file_out.create_dataset('ess', data=effective_sample)
        file_out.create_dataset('n_accepted', data=n_accepted.sum())
        file_out.create_dataset('n_total', data=n_total.sum())

        # file_out.create_dataset('individual_fit', data=individuals)
        #file_out.create_dataset('av_fc_fit', data=average_fc_fit)
        file_out.close()



    def calc_weights(self, t_prev, t_curr, tau_2, w_old, prior="None"):
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
                    norm[j] = stats.multivariate_normal.pdf(t_curr[:, i], mean=t_prev[:, j], cov=tau_2)


                weights_new[i] = p/sum(w_old * norm)

            return weights_new/weights_new.sum()

    def weighted_covar(self, x, w):
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

    def effective_sample_size(self, w):
        """
        Calculates effective sample size
        :param w: array-like importance sampleing weights
        :return: float, effective sample size
        """
        sumw = sum(w)
        sum2 = sum (w**2)
        return sumw*sumw/sum2
