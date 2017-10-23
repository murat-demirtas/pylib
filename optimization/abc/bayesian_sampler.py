"""
Module for Approximate Bayesian Computation

"""
import numpy as np
from scipy import stats
import models.dmf.core.models as dmf_model
from tools.linalg import subdiag

class Model():
    def __init__(self, sc, data, myelin=None, thickness=None, gradient_dir='forward', heterogeneity=True, min_samples=10):
        self.min_samples = min_samples
        self.heterogenous = heterogeneity
        self.twohemi = False

        if isinstance(sc, list):
            self.twohemi = True
            self.set_data(data)
            if not heterogeneity:
                self.dmf_l = dmf_model.Model(sc[0], G=1.0, myelin=None, thickness=None, verbose=False, norm_sc=True)
                self.dmf_r = dmf_model.Model(sc[1], G=1.0, myelin=None, thickness=None, verbose=False, norm_sc=True)
            else:
                self.dmf_l = dmf_model.Model(sc[0], G=1.0, myelin=myelin[0], thickness=thickness[0], gradient=gradient_dir, verbose=False, norm_sc=True)
                self.dmf_r = dmf_model.Model(sc[1], G=1.0, myelin=myelin[1], thickness=thickness[1], gradient=gradient_dir, verbose=False, norm_sc=True)
            self.dmf_l._w_EE = 1.0
            self.dmf_r._w_EE = 1.0
        else:
            self.set_data(data)
            self.dmf = dmf_model.Model(sc, G=1.0, myelin=myelin, thickness=thickness, gradient=gradient_dir, verbose=False, norm_sc=True)
            self.dmf._w_EE = 1.0

    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    def generate_data(self, theta):
        if self.twohemi:
            if self.heterogenous:
                self.dmf_l.ei_scale = (theta[0], theta[1])
                self.dmf_l.rec_scale = (theta[2], theta[3])
                self.dmf_l.G = theta[4]

                self.dmf_r.ei_scale = (theta[0], theta[1])
                self.dmf_r.rec_scale = (theta[2], theta[3])
                self.dmf_r.G = theta[5]
            else:
                self.dmf_l._J_NMDA_EI = theta[0]
                self.dmf_l._J_NMDA_rec = theta[1]
                self.dmf_l.G = theta[2]

                self.dmf_r._J_NMDA_EI = theta[0]
                self.dmf_r._J_NMDA_rec = theta[1]
                self.dmf_r.G = theta[3]

            self.dmf_l._update_matrices()
            self.dmf_r._update_matrices()
            if (self.dmf_l._unstable | self.dmf_r._unstable):
                return 1.0
            else:
                self.dmf_l._linearized_cov(BOLD=True)
                self.dmf_r._linearized_cov(BOLD=True)
                self.dmf_l.reset_state()
                self.dmf_r.reset_state()

                return [self.dmf_l.corr_bold, self.dmf_r.corr_bold]
        else:
            if self.heterogenous:
                self.dmf.ei_scale = (theta[0], theta[1])
                self.dmf.rec_scale = (theta[2], theta[3])
                self.dmf.G = theta[4]
            else:
                self.dmf._J_NMDA_EI = theta[0]
                self.dmf._J_NMDA_rec = theta[1]
                self.dmf.G = theta[2]

            self.dmf.moments_method(BOLD=True)

            if self.dmf._unstable:
                return 1.0
            else:
                return self.dmf.corr_bold

    def summary_stats(self, data):
        if self.twohemi:
            return np.hstack((subdiag(data[0]), subdiag(data[1])))
        else:
            return subdiag(data)

    def vcorrcoef(self, X, y):
        Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
        ym = np.mean(y)
        r_num = np.sum((X - Xm) * (y - ym), axis=1)
        r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
        r = r_num / r_den
        return r

    def distance_function(self, data, synth_data):
        return self.vcorrcoef(data, synth_data)

    def set_data(self, data):
        self.data = data
        self.data_sum_stats = self.data.T

    def set_prior(self, prior):
        self.prior = prior

    def generate_data_and_reduce(self, theta):
        """
        A combined method for generating data, calculating summary statistics
        and evaluating the distance function all at once.
        """
        synth = self.generate_data(theta)
        sum_stats = self.summary_stats(synth)
        d = self.distance_function(self.data_sum_stats, sum_stats)

        return d

    def run_sampler(self, epsilon, first_sample=True, theta_prev=None, weights=None, tau_squared=None):
        posterior, distances = [], []
        fit_individual = []
        fit_average_fc = []
        trial_count, accepted_count = 0, 0
        while accepted_count < self.min_samples:
            if first_sample:
                synthetic_data = 1.0
                while np.isscalar(synthetic_data):
                    theta = self.draw_theta()
                    if not (np.array(theta) < 0).any():
                        synthetic_data = self.generate_data(theta)

                    '''
                    if self.heterogenous:
                        if self.twohemi:
                            if not (np.array(theta)[[0, 2, 4, 5]] < 0).any():
                                synthetic_data = self.generate_data(theta)
                        else:
                            if not (np.array(theta)[[0, 2, 4]] < 0).any():
                                synthetic_data = self.generate_data(theta)
                    else:
                        if not (np.array(theta) < 0).any():
                            synthetic_data = self.generate_data(theta)
                    '''

                synthetic_summary_stats = self.summary_stats(synthetic_data)
                distance_ind = self.distance_function(self.data_sum_stats, synthetic_summary_stats)
                penalty = (self.data_sum_stats.mean() - synthetic_summary_stats.mean())**2
                av_fit = np.corrcoef(self.data_sum_stats.mean(0), synthetic_summary_stats)[0, 1]
                trial_count += 1
            else:
                theta_star = theta_prev[:, np.random.choice(xrange(0, theta_prev.shape[1]),
                                                            replace=True, p=weights / weights.sum())]

                synthetic_data = 1.0
                while np.isscalar(synthetic_data):
                    theta = stats.multivariate_normal.rvs(theta_star, tau_squared)
                    if not (np.array(theta) < 0).any():
                        synthetic_data = self.generate_data(theta)
                    '''
                    if self.heterogenous:
                        if self.twohemi:
                            if not (np.array(theta)[[0, 2, 4, 5]] < 0).any():
                                synthetic_data = self.generate_data(theta)
                        else:
                            if not (np.array(theta)[[0, 2, 4]] < 0).any():
                                synthetic_data = self.generate_data(theta)
                    else:
                        if not (np.array(theta) < 0).any():
                            synthetic_data = self.generate_data(theta)
                    '''

                synthetic_summary_stats = self.summary_stats(synthetic_data)
                distance_ind = self.distance_function(self.data_sum_stats, synthetic_summary_stats)
                penalty = (self.data_sum_stats.mean() - synthetic_summary_stats.mean()) ** 2
                av_fit = np.corrcoef(self.data_sum_stats.mean(0), synthetic_summary_stats)[0,1]
                trial_count += 1

            distance = 1.0 - (np.mean(distance_ind) - penalty)

            if distance < epsilon:
                accepted_count += 1
                posterior.append(theta)
                distances.append(distance)
                fit_individual.append(distance_ind)
                fit_average_fc.append(av_fit)
            else:
                pass

        posterior = np.asarray(posterior).T
        distances = np.asarray(distances)
        ind_fit = np.asarray(fit_individual)
        fit_av = np.asarray(fit_average_fc)

        return (posterior, distances,
                accepted_count, trial_count, ind_fit, fit_av)


