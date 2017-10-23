
import numpy as np
from numpy.random import randn
from math import sqrt, pi
from sys import maxint
from scipy.linalg import solve_lyapunov, eig

class HopfModel():
    def __init__(self, SC, f_diff, a):
        self.nc = SC.shape[0]
        self._z = 0.1 * np.ones((self.nc, 2))

        self._w = 0.2
        self._wC = self._w * SC
        self._sumC = np.tile(np.sum(self._wC, axis=1),(2,1)).transpose()

        self._f_diff = f_diff
        self._omega = np.tile(2 * pi * self._f_diff,(2,1)).transpose()
        self._omega[:,1] = -self._omega[:,1]
        #import pdb; pdb.set_trace()
        self._a = a * np.ones((self.nc,2))

    def _step(self, dt, dsig):
        #import pdb; pdb.set_trace()
        suma = np.dot(self._wC, self._z) - self._sumC * self._z
        self._z += dt*(self._a * self._z + \
                      self._z[:, ::-1] * self._omega - \
                      self._z * (self._z * self._z + self._z[:, ::-1] * self._z[:, ::-1]) + suma) + \
             dsig * randn(self.nc, 2)

    def integrate(self, t=30., dt=0.1, fs=2, sigma=0.02, sim_seed=None):
        #sim_seed = np.random.randint(0, maxint) if sim_seed is None else sim_seed
        #np.random.seed(sim_seed)

        # Simulation parameters
        #dt_save = dt * fs
        n_sim_steps = int(t / dt + 1)
        n_save_steps = int(t / fs + 1)
        dfs = int(fs / dt)
        dsig = sigma * sqrt(dt)
        # Synaptic state record
        synaptic_state = np.zeros((self.nc, n_save_steps))
        synaptic_state[:, 0] = self._z[:,0]

        # Main for-loop
        for i in range(1, n_sim_steps):
            self._step(dt, dsig)
            # Update state variables
            if not (i % dfs):
                #import pdb; pdb.set_trace()
                i_save = int(i / dfs)
                #print i_save
                synaptic_state[:, i_save] = self._z[:,0]
        #import pdb;
        #pdb.set_trace()
        self.t_points = np.linspace(0, t, n_save_steps)
        self.seed = sim_seed
        self.ts = synaptic_state
        return


    def _compute_jacobian(self):
        Axx = self._wC + np.diag(self._a[:,0] - self._sumC[:,0])
        Axy = np.diag(self._omega[:, 1])
        Ayx = np.diag(self._omega[:, 0])

        # Stack blocks to form full Jacobian
        col1 = np.vstack((Axx, Ayx))
        col2 = np.vstack((Axy, Axx))
        A = np.hstack((col1, col2))

        return A

    def _specturum(self, sigma=0.02):
        # Noise matrix for Lyapunov equation
        self._Q = np.identity(2 * self.nc) * sigma * sigma
        # Solve for system Jacobian
        self._jacobian = self._compute_jacobian()

        freqs = np.arange(0.001, 0.1, 0.001)
        powe = np.zeros((self.nc, len(freqs)))
        for ii,ww in enumerate(freqs):
            #import pdb; pdb.set_trace()
            value = sigma * sigma * np.dot(np.linalg.inv(self._jacobian + 1j*ww), np.linalg.inv(self._jacobian.T - 1j*ww))
            powe[:,ii] = abs(np.diag(value[:78,:78]))


        #value = sigma * sigma * np.dot(np.linalg.inv(self._jacobian + 1j * ww), np.linalg.inv(self._jacobian.T - 1j * ww))
        #powe[:, ii] = abs(np.diag(value[:78, :78]))

        # TODO: add phase coherence
        return powe
        #import pdb; pdb.set_trace()




    def _linearized_cov(self, sigma=0.02, use_lyapunov=True):
        # Solves for the linearized covariance matrix, using either
        # the Lyapunov equation or eigen-decomposition.

        # Noise matrix for Lyapunov equation
        self._Q = np.identity(2 * self.nc) * sigma * sigma

        # Solve for system Jacobian
        self._jacobian = self._compute_jacobian()

        # Eigenvalues of Jacobian matrix
        evals, L = eig(self._jacobian)

        # Check stability using eigenvalues
        self._unstable = False
        eval_is_pos = np.real(evals) > 0
        is_unstable = np.any(eval_is_pos)

        if is_unstable:
            self._unstable = True
            print "System unstable - no solution to Lyapunov equation - exiting"
            self._cov = None
            return

        if use_lyapunov:  # use Lyapunov equation to solve
            self._cov = solve_lyapunov(self._jacobian, -self._Q)

        else:  # use eigen-decomposition to solve

            evals_cc = np.conj(evals)
            L_dagger = np.conj(L).T
            inv_L = np.linalg.inv(L)
            inv_L_dagger = np.linalg.inv(L_dagger)
            Q_tilde = inv_L.dot(self._Q.dot(inv_L_dagger))
            denom_lambda_i = np.tile(evals.reshape((1, 2 * self.nc)).T,
                                     (1, 2 * self.nc))
            denom_lambda_conj_j = np.tile(evals_cc, (2 * self.nc, 1))
            total_denom = denom_lambda_i + denom_lambda_conj_j
            M = -Q_tilde / total_denom
            self._cov = L.dot(M.dot(L_dagger))

        # From covariance, get correlation
        # self._corr = cov_to_corr(self._cov)

        return self._cov[:self.nc, :self.nc]

    """
    @property
    def we(self):
        #Global coupling strength.
        return self._we

    @G.setter
    def we(self, we):
        #Global coupling strength.
        self._we = we
        return
    """