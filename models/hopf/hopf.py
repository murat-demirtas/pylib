import numpy as np
from numpy.random import randn
from math import sqrt, pi
from scipy.linalg import solve_lyapunov, eig

class HopfModel():
    def __init__(self, SC, f_diff, a = 0.0, hmap = None, g = 0.0, sigma = 0.02):
        self.nc = SC.shape[0]
        self._z = 0.1 * np.ones((self.nc, 2))

        self._SC = SC

        self._w = g
        self._wC = self._w * self._SC
        self._sumC = np.tile(np.sum(self._wC, axis=1),(2,1)).T

        self._f_diff = f_diff

        self._omega = np.tile(2 * pi * self._f_diff,(2,1)).T
        self._omega[:,1] = -self._omega[:,1]

        self._sigma = sigma

        self._synaptic_state = None

        # Heterogeneity map values for each area
        self._raw_hmap = hmap
        self._hamp = 0.0
        self._hmap_rev = 0.0

        # Set heterogeneity gradients
        if self._raw_hmap is not None:
            hmap_range = np.ptp(self._raw_hmap)
            self._hmap = (-(self._raw_hmap - np.max(self._raw_hmap)) / hmap_range)

            hmap_norm = self._raw_hmap - np.min(self._raw_hmap)
            self._hmap_rev = hmap_norm / np.max(hmap_norm)

        self.a = a

    def _step(self, dt, sigma_dt):
        suma = np.dot(self._wC, self._z) - self._sumC * self._z
        self._z += dt * (self._a * self._z
                         + self._z[:, ::-1] * self._omega
                         - self._z * (self._z * self._z + self._z[:, ::-1] * self._z[:, ::-1])
                         + suma) \
                         + sigma_dt * randn(self.nc, 2)



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

    def _apply_hierarchy(self, a, b):
        """
        Parametrize model parameters based on the heterogeneity map.

        Parameters
        ----------
        a : float
            The interceot term
        b : float
            The scaling factor

        Returns
        -------
        ndarray
            Parameter values varying along heterogeneity gradient given the intercept and scaling
            factor.

        Notes
        -----
        If b is negative the heterogeneity gradient will be calculated in opposite direction.
        """
        if b < 0.0:
            return a + np.abs(b) * self._hmap_rev
        else:
            return a + b * self._hmap

    def moments_method(self, use_lyapunov=True):
        # Solves for the linearized covariance matrix, using either
        # the Lyapunov equation or eigen-decomposition.

        # Noise matrix for Lyapunov equation
        self._Q = np.identity(2 * self.nc) * self._sigma * self._sigma

        # Solve for system Jacobian
        self._jacobian = self._compute_jacobian()

        # Eigenvalues of Jacobian matrix
        evals, L = eig(self._jacobian)

        self.evals = evals

        # Check stability using eigenvalues
        self._unstable = False
        eval_is_pos = np.real(evals) > 0
        is_unstable = np.any(eval_is_pos)

        if is_unstable:
            self._unstable = True
            self._cov = None
            return

        if use_lyapunov:  # use Lyapunov equation to solve
            self._cov = solve_lyapunov(self._jacobian, -self._Q)
        else:
            # use eigen-decomposition to solve
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
            self._cov = L.dot(M.dot(L_dagger)).real

    def integrate(self, t=30., dt=0.1, fs=2):
        # Simulation parameters
        n_sim_steps = int(t / dt + 1)
        n_save_steps = int(t / fs + 1)
        fs_dt = int(fs / dt)
        sigma_dt = self._sigma * sqrt(dt)

        # Synaptic state record
        self._t = np.linspace(0, t, n_save_steps)
        self._y = np.zeros((self.nc, n_save_steps))
        self._y[:, 0] = self._z[:, 0]

        # Main for-loop
        for i in range(1, n_sim_steps):
            self._step(dt, sigma_dt)

            # Update state variables
            if not (i % fs_dt):
                i_save = int(i / fs_dt)
                self._y[:, i_save] = self._z[:, 0]

    @property
    def y(self):
        return self._y

    @property
    def cov(self):
        return self._cov[:self.nc, :self.nc]

    @property
    def corr(self):
        cov_ii = np.diag(self.cov)
        norm2 = np.outer(cov_ii, cov_ii)
        return self.cov / np.sqrt(norm2)

    @property
    def g(self):
        #Global coupling strength.
        return self._w

    @g.setter
    def g(self, g):
        self._w = g
        self._wC = self._w * self._SC
        self._sumC = np.tile(np.sum(self._wC, axis=1),(2,1)).T

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        """
        a : ndarray
            Local bifurcation parameters

        Notes
        -----
        If w is float, sets all strengths to w;
        if w has N elements (number of regions), sets all strengths to w;
        if w has size 2, sets w according to heterogeneity map
        """
        if isinstance(a, float):
            self._a = a * np.ones((self.nc,2))
        else:
            if len(a) == self.nc:
                self._a = np.tile(a,(2,1)).T
            else:
                self._a = np.tile(self._apply_hierarchy(a[0], a[1]), (2, 1)).T