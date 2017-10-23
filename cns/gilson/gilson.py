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

        # Optimization Parameters:
        for ii in range(self.N): self.SC[ii,ii] = 0.0 # Remove diagonal
        self.mask = np.where(self.SC > SC_thr) # non-zero index mask
        self.mask_diag = np.eye(self.N, dtype=bool) # diagonal mask
        self.tau = tau # time shift for fc_tau

        # Constraints for EC and Sigma
        self.max_EC = maxEC  # maximal value for EC
        self.min_sigma = 0.01  # minimal value for Sigma

        # Optimization step size
        self.epsilon_EC = 0.0005 # optimization step for EC
        self.epsilon_sigma = 1.0 # optimization step for sigma

        # Dummy variable initializations
        self.fc_0_obj = None
        self.fc_tau_obj = None

    def set_empirical_objectives(self, TS=None, tau_est=True, fc_norm=0.5):
        ## Computes objectives (fc-0 and fc-tau) for given time series
        if TS is not None:
            TS -= np.outer(np.ones(TS.shape[1]), TS.mean(1)).T
            self.fc_0_obj = np.cov(TS)
            self.fc_tau_obj = self._xcov(TS, self.tau)
            norm_factor = self.fc_0_obj.mean()
            self.fc_0_obj *= fc_norm / self.fc_0_obj.mean()
            self.fc_tau_obj = fc_norm * self.fc_tau_obj / norm_factor
        if tau_est: self._get_taux()

        ## Initialize optimization parameters
        self.coef_0 = np.sqrt(np.sum(self.fc_tau_obj ** 2)) / \
                      (np.sqrt(np.sum(self.fc_0_obj ** 2)) + np.sqrt(np.sum(self.fc_tau_obj ** 2)))
        self.coef_tau = 1.0 - self.coef_0

    def optimize(self, n_opt, random=False, tolerance=100):
        ## Lyapunov Optimization
        if self.fc_0_obj is None:
            raise Exception('cannot optimize: define objective variables first...')

        t0 = time()
        self._reset(random) # initialize parameters
        self.fc_dist_mem = np.zeros(n_opt)
        self.fc_fit_mem = np.zeros(n_opt)
        stop_opt = False
        count = 0
        while not stop_opt:
            self._linearized_cov() # compute linearized covariance
            self._compute_error() # compute error

            fc_distance = 0.5 * (self.err_fc_0 + self.err_fc_tau)
            pearson_fit = 0.5 * (pearsonr(self.fc_0.reshape(-1), self.fc_0_obj.reshape(-1))[0] +
                                 pearsonr(self.fc_tau.reshape(-1), self.fc_tau_obj.reshape(-1))[0])

            if fc_distance < self.best_fit:
                # store best fit
                self.best_fit = fc_distance
                self.best_EC = copy(self.EC)
                self.best_sigma = copy(np.diag(self.sigma))
                self.best_fc_0 = copy(self.fc_0)
                self.best_fc_tau = copy(self.fc_tau)
            else:
                if count > tolerance: stop_opt = True

            if count%int(n_opt/10) == 0: self._report_time(count, n_opt, t0, pearson_fit)

            self._update_jacobian()  # compute Jacobian update
            self._update_parameters() # update EC and sigma

            # Store fitting values
            self.fc_dist_mem[count] = fc_distance
            self.fc_fit_mem[count] = pearson_fit
            count += 1

            # Stop criteria
            if count == n_opt: stop_opt = True

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
    def _get_dist(self, C):
        # Compute distance for single step
        J = -np.eye(self.N) / self.tau_x + C
        fc_0 = solve_lyapunov(J, -self.sigma)
        fc_tau = np.dot(fc_0, expm(J.T * self.tau))
        err_fc_0 = self._norm_dist(fc_0, self.fc_0_obj)
        err_fc_tau = self._norm_dist(fc_tau, self.fc_tau_obj)
        return 0.5 * (err_fc_0 + err_fc_tau)

    def _compute_jacobian(self, C):
        ## Compute Jacobian
        return -np.eye(self.N) / self.tau_x + C

    def _linearized_cov(self):
        ## Approximated FC-0 and FC-tau
        self.J = self._compute_jacobian(self.EC)
        self.fc_0 = solve_lyapunov(self.J, -self.sigma)
        self.fc_tau = np.dot(self.fc_0, expm(self.J.T * self.tau))

    def _compute_error(self):
        ## Compute error
        self.err_fc_0 = self._norm_dist(self.fc_0, self.fc_0_obj)
        self.err_fc_tau = self._norm_dist(self.fc_tau, self.fc_tau_obj)

    def _update_jacobian(self):
        ## Jacobian update with weighted FC updates depending on respective error
        exp_term = expm(-self.J.T * self.tau)

        # Error in FC
        delta_fc_0 = (self.fc_0_obj - self.fc_0) * self.coef_0
        delta_fc_tau = (self.fc_tau_obj - self.fc_tau) * self.coef_tau

        # Jacobian and sigma update coefficients
        self.delta_J = np.dot(np.linalg.pinv(self.fc_0),
                              delta_fc_0 + np.dot(delta_fc_tau, exp_term)).T / self.tau
        self.delta_sigma = np.dot(self.J, delta_fc_0) + np.dot(delta_fc_0, self.J.T)

    def _update_parameters(self):
        self.EC[self.mask] += self.delta_J[self.mask] * self.epsilon_EC
        self.EC = np.clip(self.EC, 0.0, self.max_EC)

        self.sigma[self.mask_diag] -= self.epsilon_sigma * self.delta_sigma[self.mask_diag]
        self.sigma = np.clip(self.sigma, self.min_sigma, np.inf)

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


class ArtificialLM(Linear):
    ## Artificial Network Model
    def __init__(self, SC=50, sigma=0.5, tau=2, pd=0.2, wmax=1.0, fixed_sigma=False):
        ## If SC is an integer, generates random matrix with
        ## corresponding dimension. i.e. NxN matrix...
        ## pd: probability density, wmax: maximum weight
        if isinstance(SC, int):
            w_C = wmax
            SC = self.gen_random_C(SC, pd, w_C)

        if not fixed_sigma: sigma += 0.1 * np.random.randn(SC.shape[0])
        self.sigma_original = sigma ** 2 * np.eye(SC.shape[0])

        super(ArtificialLM, self).__init__(SC=SC, tau=tau, SC_thr=0.0, maxEC=wmax)

    def set_simulated_objectives(self, nsim=50, tsim=300, dt=0.1, t0=200):
        ## Compute objectives using numerical simulations
        sigma = np.diag(np.sqrt(self.sigma_original))
        self.x = np.random.random(self.N)
        self.fc_0_obj = np.zeros((self.N, self.N))
        self.fc_tau_obj = np.zeros((self.N, self.N))
        for jj in range(nsim):
            TS = self.integrate(self.SC, sigma_sim=sigma, tsim=tsim, dt=dt, t0=t0)
            self.fc_0_obj += np.cov(TS)
            self.fc_tau_obj += self._xcov(TS, np.floor(self.tau/dt))
        self.fc_0_obj /= nsim
        self.fc_tau_obj /= nsim

        ## Initialize optimization parameters
        self.coef_0 = np.sqrt(np.sum(self.fc_tau_obj ** 2)) / \
                      (np.sqrt(np.sum(self.fc_0_obj ** 2)) + np.sqrt(np.sum(self.fc_tau_obj ** 2)))
        self.coef_tau = 1.0 - self.coef_0

    def set_theoretical_objectives(self):
        ## compute objectives using theoretical values
        Q = self.fc_theoretical(self.SC, self.tau)
        self.fc_0_obj = Q[0]
        self.fc_tau_obj = Q[1]

        ## Initialize optimization parameters
        self.coef_0 = np.sqrt(np.sum(self.fc_tau_obj ** 2)) / \
                      (np.sqrt(np.sum(self.fc_0_obj ** 2)) + np.sqrt(np.sum(self.fc_tau_obj ** 2)))
        self.coef_tau = 1.0 - self.coef_0

    def gen_random_C(self, N, p_arg, w_arg):
        ## generate random connectivity matrix
        C = np.random.rand(N, N)
        C[C > p_arg] = 0
        for ii in range(N): C[ii, ii] = 0
        return w_arg * C

    def fc_theoretical(self, C_orig, v_tau):
        ## compute theoretical objectives
        J = self._compute_jacobian(C_orig)
        qtau = np.empty((2, self.N, self.N))
        qtau[0, :, :] = solve_lyapunov(J, -self.sigma_original)
        qtau[1, :, :] = np.dot(qtau[0, :, :], expm(J.T * v_tau))
        return qtau
