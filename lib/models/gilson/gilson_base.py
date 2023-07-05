import numpy as np
from scipy.linalg import solve_lyapunov, expm
from scipy.stats import pearsonr
from time import time
from copy import copy

class Linear(object):
    def __init__(self, SC, tau, SC_thr=0.0, maxEC=1.0):
        # Model Parameters:
        self.SC = SC # structural connectivity
        self.N = SC.shape[0] # number of regions
        self.tau_x = 2.33 # time constant of dynamical system
        self.I_ext = 1.0 # external input

    def integrate(self, C, sigma_sim, tsim, dt, t0):
        x = np.random.random(self.N)
        tspan = int(tsim / dt)
        x_mem = np.empty((self.N, tspan))
        for ii in range(tspan):
            x += dt * (-x / self.tau_x + np.dot(C, x) + self.I_ext) + np.sqrt(dt) * np.random.randn(self.N) * sigma_sim
            x_mem[:, ii] = x[:]
        return x_mem[:, t0:]

    def integrate_gs(self, C, sigma_sim, tsim, dt, t0):
        x = np.random.random(self.N)
        tspan = int(tsim / dt)
        x_mem = np.empty((self.N, tspan))
        for ii in range(tspan):
            x += dt * (-x / self.tau_x + np.dot(C, x) + self.I_ext) + np.sqrt(dt) * np.random.randn(
                self.N) * sigma_sim + np.sqrt(dt) * np.random.randn() * 0.1
            x_mem[:, ii] = x[:]
        return x_mem[:, t0:]

    ######################
    ### Private Functions
    ######################
    def _compute_jacobian(self, C):
        ## Compute Jacobian
        return -np.eye(self.N) / self.tau_x + C

    def _linearized_cov(self):
        ## Approximated FC-0 and FC-tau
        self.J = self._compute_jacobian(self.EC)
        self.fc_0 = solve_lyapunov(self.J, -self.sigma)
        self.fc_tau = np.dot(self.fc_0, expm(self.J.T * self.tau))

    def _xcov(self, x, lag):
        c = np.empty((self.N, self.N))
        xi = x[:, lag:]
        yi = x[:, :-lag]
        for i in range(self.N):
            for j in range(self.N):
                c[i][j] = np.cov(xi[i,:],yi[j,:])[0,1]
        return c

    def _get_taux(self):
        v_tau = np.array([0, self.tau])
        fc_agg = np.dstack((self.fc_0_obj, self.fc_tau_obj))
        log_ac = np.log(np.maximum(fc_agg.diagonal(axis1=0,axis2=1), 1e-10))
        lin_reg = np.polyfit(np.repeat(v_tau, self.N), log_ac.reshape(-1), 1)
        self.tau_x = -1. / lin_reg[0]

    def _norm_dist(self, x, x0):
        ## Distance function
        return np.sqrt(np.sum((x - x0) ** 2) / np.sum(x0 ** 2))

    def _reset(self, init_guess):
        if init_guess:
            # if true, use random initial guess
            self.sigma = np.eye(self.N) * 0.5 + 0.1 * np.diag(np.random.randn(self.N))
            self.EC = np.random.random((self.N, self.N)) * self.max_EC
            self.EC[self.mask] = 0.0
        else:
            # use fixed valus
            self.sigma = np.eye(self.N)
            self.EC = np.zeros((self.N, self.N))

        self.best_fit = 10e10
        self.best_EC = copy(self.EC)
        self.best_sigma = copy(np.diag(self.sigma))
        self.best_fc_0 = None
        self.best_fc_tau = None
        self.delta_J = np.zeros((self.N, self.N))
        self.delta_sigma = np.zeros(self.N)

    def _report_time(self, current, total, t0, fit):
        t1 = time()
        print('Best Fit: {2:.2f} - {0:d}% complete, time elapsed: {1:.2f} s.'.format(100*current/total, t1-t0, fit))

