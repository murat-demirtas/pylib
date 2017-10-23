#!/usr/bin/python

""" Dynamic mean field model base class. See below for extended model
derived from base class. """
from ..utils.general import cov_to_corr
from ..utils.load_data import load_model_params
from .hemo import Balloon
from .sim import Sim
from scipy.optimize import curve_fit, fsolve
from scipy.linalg import solve_lyapunov, eig, expm
import numpy as np
from collections import Iterable
import sympy as sym

class Model(object):

    def __init__(self, I_ext=0.0, bold='obata', params=None, verbose=False):
        # Print diagnostics to console
        self._verbose = verbose

        # Add hemodynamic model (Balloon-Windkessel)
        self.hemo = Balloon(1, parameters=bold)

        # Add simulation class
        self.sim = Sim()

        # Load model parameters
        model_params = load_model_params()
        if params is not None:
            for key in params.keys():
                model_params[key] = params[key]

        # Unstable if Jacobian has eval > 0
        self._unstable = False

        # Feedback inhibition weights
        self._J = 1.0

        # Initialize model outputs to None
        self._jacobian = None
        self._cov = None
        self._corr = None
        self._cov_bold = None
        self._corr_bold = None

        # Initialize state members to None
        self._I_E = None
        self._I_I = None
        self._S_E = None
        self._S_I = None
        self._r_E = None
        self._r_I = None

        # Various model parameters
        self._w_II = model_params['w_II']
        self._w_IE = model_params['w_IE']
        self._I0 = model_params['I0']
        self._J_NMDA_rec = model_params['J_NMDA']
        self._J_NMDA_EE = model_params['J_NMDA']
        self._J_NMDA_EI = model_params['J_NMDA']
        self._sigma = model_params['sigma']
        self._gamma = model_params['gamma']
        self._W_I = model_params['W_I']
        self._W_E = model_params['W_E']
        self._w_EE = model_params['w_EE']
        self._tau_I = model_params['tau_I']
        self._tau_E = model_params['tau_E']
        self._d_I = model_params['d_I']
        self._d_E = model_params['d_E']
        self._b_I = model_params['b_I']
        self._b_E = model_params['b_E']
        self._a_I = model_params['a_I']
        self._a_E = model_params['a_E']
        self._I_ext = I_ext

        self._I0_E = self._W_E * self._I0
        self._I0_I = self._W_I * self._I0

        # Steady state values for isolated node
        self._I_E_ss = model_params['I_E_ss']
        self._I_I_ss = model_params['I_I_ss']
        self._S_E_ss = model_params['S_E_ss']
        self._S_I_ss = model_params['S_I_ss']
        self._r_E_ss = model_params['r_E_ss']
        self._r_I_ss = model_params['r_I_ss']

        # Noise covariance matrix
        self._Q = np.identity(2) * self._sigma * self._sigma

        # Add lookup tables for transfer function and its derivatives
        self._phi()

        return

    def __repr__(self):
        return "dynamic mean field model class"

    def __str__(self):
        msg = ""
        msg += '%-17s %s' % ("\nUnstable:", self._unstable)
        msg += '\n%-16s %s' % ("Coupling (G):", self._G)
        msg += '\n%-16s %s' % ("N areas:", self._nc)
        return msg

    """ Auxiliary Functions"""

    def _phi(self):
        """Generate transfer function for Excitatory and Inhibitory populations"""
        IE = sym.symbols('IE')
        II = sym.symbols('II')
        phi_E = (self._a_E * IE - self._b_E) / (1. - sym.exp(-self._d_E * (self._a_E * IE - self._b_E)))
        phi_I = (self._a_I * II - self._b_I) / (1. - sym.exp(-self._d_I * (self._a_I * II - self._b_I)))
        dphi_E = sym.diff(phi_E, IE)
        dphi_I = sym.diff(phi_I, II)

        self.phi_E = sym.lambdify(IE, phi_E)
        self.phi_I = sym.lambdify(II, phi_I)
        self.dphi_E = sym.lambdify(IE, dphi_E)
        self.dphi_I = sym.lambdify(II, dphi_I)

        return

    def _update_matrices(self, compute_FIC=True):

        # Connectivity
        self._K_EE = self._J_NMDA_rec * self._w_EE
        self._K_EI = self._J_NMDA_EI
        if compute_FIC: self._J = self._analytic_FIC()
        self._K_IE = -self._J
        self._K_II = -1.0

        # Derivatives of transfer function for each cell type
        # at steady state value of current
        dr_E = self.dphi_E(self._I_E_ss)
        dr_I = self.dphi_I(self._I_I_ss)

        # A_{mn} = dS_i^m/dS_j^n
        A_EE = (-1. / self._tau_E - (self._gamma * self._r_E_ss)) + \
               ((-self._gamma * (self._S_E_ss - 1.)))*(dr_E*(self._K_EE))
        A_IE = ((self._gamma * (1. - self._S_E_ss)))*(dr_E*(self._K_IE))
        A_EI = dr_I*(self._K_EI)
        A_II = (-1. / self._tau_I) + dr_I*(self._K_II)

        # Stack blocks to form full Jacobian
        col1 = np.vstack((A_EE, A_EI))
        col2 = np.vstack((A_IE, A_II))
        self._jacobian = np.hstack((col1, col2))

        # Eigenvalues of Jacobian matrix
        self._evals, self._evects = eig(self._jacobian)

        self._max_eval = np.real(self._evals.max())
        # Check stability using eigenvalues
        self._unstable = self._max_eval >= 0.0

    def _inh_curr_fixed_pts(self, I):
        """ Auxiliary function to find steady state inhibitory
        currents when FFI is enabled."""
        return self._I0_I + self._K_EI*self._S_E_ss + self._I_ext - \
            self._w_II * self._tau_I * self.phi_I(I) - I

    def _analytic_FIC(self):
        """ Analytically solves for the strength of feedback
            inhibition for each cortical area. """
        # Numerically solve for inhibitory currents first
        I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts,
                                             x0=self._I_I_ss, full_output=True)

        if ier:  # successful exit from fsolve
            self._I_I = np.copy(I_I_ss)           # needed for self._update_rate_I()
            self._I_I_ss = np.copy(I_I_ss)        # update stored steady state value
            self._r_I = self.phi_I(self._I_I)     # compute new steady state rate
            self._r_I_ss = np.copy(self._r_I)     # update stored steady state value
            self._S_I_ss = np.copy(self._r_I_ss) * self._tau_I  # update stored val.
        else:
            err_msg = "Failed to find new steady-state currents." + \
                      " Cause of failure: %s" % mesg
            raise Exception(err_msg)

        # Solve for J using the steady state values (fixed points)
        J = (-1. / self._S_I_ss) * \
            (self._I_E_ss -
             self._I_ext - self._I0_E -
             self._K_EE*self._S_E_ss)

        if np.any(J < 0):
            raise ValueError("FIC calculation led to negative J values!")
        return J

    def _solve_lyapunov(self, jacobian, evals, L, Q, builtin=False):
        if builtin:
            return solve_lyapunov(jacobian, -Q)
        else:
            n = jacobian.shape[0] / 2
            evals_cc = np.conj(evals)
            L_dagger = np.conj(L).T
            inv_L_dagger = np.linalg.inv(L_dagger)
            Q_tilde = np.linalg.solve(L, Q.dot(inv_L_dagger))
            denom_lambda_i = np.tile(evals.reshape((1, 2 * n)).T,
                                     (1, 2 * n))
            denom_lambda_conj_j = np.tile(evals_cc, (2 * n, 1))
            total_denom = denom_lambda_i + denom_lambda_conj_j
            M = -Q_tilde / total_denom
            return L.dot(M.dot(L_dagger)).real

    def _linearized_cov(self, use_lyapunov=False, BOLD=False):
        """ Solves for the linearized covariance matrix, using either
        the Lyapunov equation or eigen-decomposition."""
        if self._unstable:
            if self._verbose: print "System unstable - no solution to Lyapunov equation - exiting"
            self._cov, self._cov_bold, self._corr_bold, self._corr = None, None, None, None
            return
        else:
            if BOLD:
                self.hemo.linearize_BOLD(self._S_E_ss, self._jacobian, self._Q)
                self._jacobian_bold = self.hemo.full_A
                self._Q_bold = self.hemo.full_Q
                evals, evects = eig(self._jacobian_bold)
                self._full_cov = self._solve_lyapunov(self._jacobian_bold, evals, evects, self._Q_bold, builtin=use_lyapunov)
                self._cov = self._full_cov[:2, :2]
                self._cov_bold = np.dot(np.dot(self.hemo.B, self._full_cov), self.hemo.B.conj().T)
                self._corr = cov_to_corr(self._cov, full_matrix=True)
                self._corr_bold = cov_to_corr(self._cov_bold, full_matrix=False)
            else:
                self._cov = self._solve_lyapunov(self._jacobian, self._evals, self._evects, self._Q, builtin=use_lyapunov)
                self._corr = cov_to_corr(self._cov)

    """ Approximation Functions"""
    def power_spectrum(self, freqs, tau=0.0):
        """Returns the power in each cortical area at the given frequencies
        for the specified neuronal population ('E' or 'I'). """
        if self._jacobian is None: self._update_matrices()

        Id = np.identity(2)
        power = np.empty((2, 2, len(freqs)), dtype=complex)
        sig = complex(self._sigma ** 2)
        for i, f in enumerate(freqs):
            w = 2. * np.pi * f
            M1 = np.linalg.inv(self.jacobian + 1.j * w * Id)
            M2 = np.linalg.inv(self.jacobian.T - 1.j * w * Id)
            M3 = np.dot(M1, M2)
            if tau == 0:
                power[:,:,i] = M3 * sig
            else:
                power[:, :, i] = sig * M3.dot(expm(self.jacobian.T*tau))
        return power

    def coherence(self, freqs, tau=0.0):
        density = np.abs(self.power_spectrum(freqs,tau=tau)).real
        if density.ndim > 2: density = density.sum(2)
        power = np.diag(density)
        power_diag = np.tile(power, (2, 1))
        coh = (density ** 2) / (power_diag * power_diag.T)
        return coh, power

    def moments_method(self, recompute_FIC=True, BOLD=False, use_lyapunov=False):
        self._update_matrices(compute_FIC=recompute_FIC)
        self._linearized_cov(use_lyapunov=use_lyapunov, BOLD=BOLD)
        self.reset_state()

        return

    """ Numerical Simulations """
    def exc_current(self):
        """ Excitatory current for each cortical area. """
        return self._I0_E + self._I_ext + self._K_EE * self._S_E + self._K_IE * self._S_I

    def inh_current(self):
        """ Inhibitory current for each cortical area. """
        return self._I0_I + self._K_EI * self._S_E + self._K_II * self._S_I

    def _dSEdt(self):
        """Returns time derivative of excitatory synaptic gating variables
        in absense of noise."""
        return -(self._S_E / self._tau_E) + (self._gamma * self._r_E) * (1. - self._S_E)

    def _dSIdt(self):
        """Returns time derivative of inhibitory synaptic gating variables
        in absense of noise."""
        return -(self._S_I / self._tau_I) + self._r_I

    def reset_state(self):
        """Reset state members to steady-state values."""

        self._I_I = np.copy(self._I_I_ss)
        self._I_E = np.copy(self._I_E_ss)
        self._r_I = np.copy(self._r_I_ss)
        self._r_E = np.copy(self._r_E_ss)
        self._S_I = np.copy(self._S_I_ss)
        self._S_E = np.copy(self._S_E_ss)

        return

    def step(self, dt):
        """ Advance system synaptic state by time evolving for time dt. """

        # Update currents
        self._I_E = self.exc_current()
        self._I_I = self.inh_current()

        self._r_E = self.phi_E(self._I_E)
        self._r_I = self.phi_I(self._I_I)

        # Compute change in synaptic gating variables
        dS_E = self._dSEdt() * dt + np.sqrt(dt) * self._sigma * \
               np.random.normal(size=2)

        dS_I = self._dSIdt() * dt + np.sqrt(dt) * self._sigma * \
               np.random.normal(size=2)

        # Update class members S_E, S_I
        self._S_E += dS_E
        self._S_I += dS_I

        # Clip synaptic gating fractions
        self._S_E = np.clip(self._S_E, 0., 1.)
        self._S_I = np.clip(self._S_I, 0., 1.)

        return

    def integrate(self, t=30., dt=1e-4, n_save=10, include_BOLD=True,
                  from_fixed=True, sim_seed=None):

        """ Time evolves the system using Euler integration.

        :param t: simulation length (in seconds)
        :param dt: simulation time step (in seconds)
        :param n_save: time steps between recordings
        :param include_BOLD: compute BOLD outputs as well
        :param from_fixed: start system from fixed point
        :param save_dir: save directory for pickle containing simulation dict
        :param sim_seed: seed for random number generator

        """

        sim_seed = np.random.randint(0, 4294967295) if sim_seed is None else sim_seed
        np.random.seed(sim_seed)

        # Initialize to fixed point
        if from_fixed:
            self.reset_state()
            self.hemo.reset_state()

        # Simulation parameters
        dt_save = dt * n_save
        n_sim_steps = int(t / dt + 1)
        n_save_steps = int(t / dt_save + 1)

        # Synaptic state record
        synaptic_state = np.zeros((6, 2, n_save_steps))
        synaptic_state[:, :, 0] = self.state

        if self._verbose:
            print "Beginning simulation."

        # Initialize BOLD variables if required
        if include_BOLD:
            # Hemodynamic state record
            hemo_state = np.zeros((5, 2, n_save_steps))
            hemo_state[:3, :, 0] = 1.  # ICs

        # Main for-loop
        for i in range(1, n_sim_steps):

            self.step(dt)

            #if include_BOLD:
            #    self.hemo.step(dt, self._S_E)

            # Update state variables
            if not (i % n_save):
                i_save = i / n_save
                synaptic_state[:, :, i_save] = self.state

                if include_BOLD:
                    self.hemo.step(dt*10., self._S_E)
                    hemo_state[:, :, i_save] = self.hemo.state

                if self._verbose:
                    if not (i_save % 1000):
                        print i_save

        if self._verbose:
            print "Simulation complete."

        self.sim.t = t
        self.sim.dt = dt_save
        self.sim.n_save = n_save
        self.sim.t_points = np.linspace(0, t, n_save_steps)
        self.sim.seed = sim_seed

        self.sim.I_I, self.sim.I_E, self.sim.r_I, self.sim.r_E, \
        self.sim.S_I, self.sim.S_E = synaptic_state

        if include_BOLD:
            self.sim.x, self.sim.f, self.sim.v, self.sim.q, self.sim.y = hemo_state

        return

    def cov_tau(self, tau=1.0):
        return self._cov.dot(expm(self._jacobian.T * tau))

    def corr_tau(self, tau=1.0):
        return self.cov_tau(tau=tau) / np.sqrt(np.outer(self.var, self.var))

    def cov_bold_tau(self, tau=1.0):
        return np.dot(np.dot(self.hemo.B, self._full_cov.dot(expm(self._jacobian_bold.T * tau))), self.hemo.B.conj().T).T

    def corr_bold_tau(self, tau=1.0):
        return self.cov_bold_tau(tau=tau) / np.sqrt(np.outer(self.var_bold, self.var_bold))

    @property
    def Q(self):
        return self._Q

    @property
    def cov(self):
        """Covariance matrix of linearized fluctuations
        about fixed point."""
        return self._cov
    @property
    def var(self):
        return np.diag(self._cov)

    @property
    def var_bold(self):
        return np.diag(self._cov_bold)

    @property
    def corr(self):
        """Correlation matrix of linearized
        fluctuations about fixed point."""
        return self._corr

    @property
    def cov_bold(self):
        """Covariance matrix of linearized fluctuations
        about fixed point."""
        return self._cov_bold

    @property
    def corr_bold(self):
        """Correlation matrix of linearized
        fluctuations about fixed point."""
        return self._corr_bold


    @property
    def jacobian(self):
        """Jacobian of linearized fluctuations about fixed point."""
        return self._jacobian

    @property
    def evals(self):
        """Eigenvalues of Jacobian matrix."""
        return eig(self._jacobian)[0] if self.jacobian is not None else None

    @property
    def evecs(self):
        """Left eigenvectors of Jacobian matrix."""
        return eig(self._jacobian)[1] if self.jacobian is not None else None

    @property
    def nc(self):
        """Number of cortical areas."""
        return self._nc

    @property
    def SC(self):
        """Empirical structural connectivity."""
        return self._SC

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
        self._Q = np.identity(2) * self._sigma * self._sigma

    @property
    def state(self):
        """All state variables, 6 rows by len(nodes) columns.
            Rows are I_I, I_E, r_I, r_E, S_I, S_E. """
        return np.vstack((self._I_I, self._I_E, self._r_I,
                          self._r_E, self._S_I, self._S_E))

    @property
    def steady_state(self):
        """All steady state variables, shape 6 x nc.
            Rows are, respectively, I_I, I_E, r_I, r_E, S_I, S_E. """
        return np.vstack((self._I_I_ss, self._I_E_ss, self._r_I_ss,
                          self._r_E_ss, self._S_I_ss, self._S_E_ss))

    @property
    def w_EE(self):
        """Self-recurrence strengths for each excitatory population."""
        return self._w_EE

    @w_EE.setter
    def w_EE(self, w):
        self._w_EE = w

    @property
    def G(self):
        """Global coupling strength."""
        return self._G

    @G.setter
    def G(self, g):
        """Global coupling strength."""
        self._G = g
        return

    @property
    def J(self):
        """Feedback inhibition weights."""
        return self._J

    @J.setter
    def J(self, J_i):
        """Global coupling strength."""
        self._J = J_i
        return

    @property
    def I_ext(self):
        return self._I_ext

    @I_ext.setter
    def I_ext(self, I):
        self._I_ext = I
