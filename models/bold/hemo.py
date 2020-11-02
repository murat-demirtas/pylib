#! /usr/bin/python

""" Hemodynamic transfer function class for input-state-output. """
import numpy as np

class Balloon(object):

    def __init__(self, nc, parameters='Obata'):

        self.obata = True
        # Number of cortical areas
        self._nc = nc
        if parameters == 'Obata':
            """
            Hemodynamic parameters for Balloon-Windkessel model according to Obata 2004.
            """
            V0 = 0.02  # resting blood volume fraction
            kappa = 0.65  # [s^-1] rate of signal decay
            gamma = 0.41  # [s^-1] rate of flow-dependent elimination
            tau = 0.98  # [s] hemodynamic transit time
            alpha = 0.32  # Grubb's exponent
            rho = 0.34  # resting oxygen extraction fraction

            k1 = 7 * rho
            k2 = 1.43 * rho
            k3 = 0.43

        if parameters == 'Friston':
            """
            Hemodynamic parameters for Balloon-Windkessel model according to Friston 2003.
            """
            V0 = 0.02  # resting blood volume fraction
            kappa = 0.65  # [s^-1] rate of signal decay
            gamma = 0.41  # [s^-1] rate of flow-dependent elimination
            tau = 0.98  # [s] hemodynamic transit time
            alpha = 0.32  # Grubb's exponent
            rho = 0.34  # resting oxygen extraction fraction

            # Friston 2003 equations
            k1 = 7. * rho
            k2 = 2.
            k3 = 2. * rho - 0.2
            self.obata = False


        # Hemodynamic model parameters
        self._V0 = V0
        self._kappa = kappa
        self._gamma = gamma
        self._tau = tau
        self._alpha = alpha
        self._rho = rho
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3

        # Hemodynamic state variables
        self._x = np.ones(self._nc)   # vasodilatory signal
        self._f = np.ones(self._nc)   # normalized inflow rate
        self._v = np.ones(self._nc)   # normalized blood volume
        self._q = np.zeros(self._nc)  # normalized deoxyhemoglobin content
        self._y = np.zeros(self._nc)  # BOLD

    def BOLD_tf(self, freqs):
        """The analytic solution to the transfer function
        of the BOLD signal y as a function of the input
        synaptic signal z, at a given frequency f, for the
        Balloon-Windkessel hemodynamic model. For derivation
        details see Robinson et al., 2006, BOLD responses to
        stimuli. """
        w = 2 * np.pi * freqs
        beta = (self._rho + (1.-self._rho)*np.log(1.-self._rho)) / self._rho
        T_yz = (self._V0 * (self._alpha * (self._k2 + self._k3) * (w*self._tau*1.j - 1) +
               (self._k1 + self._k2) * (self._alpha + beta - 1. - w *
                self._alpha*beta*self._tau*1.j))) / ((1. - w * self._tau*1.j) *
               (1. - self._alpha * w * self._tau*1.j) * (w**2 + w * self._kappa*1.j -
                                                     self._gamma))
        return T_yz * np.conj(T_yz)

    def reset_state(self):
        """Reset hemodynamic state variables to steady state values. """

        self._x = np.ones(self._nc)   # vasodilatory signal
        self._f = np.ones(self._nc)   # normalized inflow rate
        self._v = np.ones(self._nc)   # normalized blood volume
        self._q = np.zeros(self._nc)  # normalized deoxyhemoglobin content
        self._y = np.zeros(self._nc)  # BOLD

        return

    def step(self, dt, x):
        """ Evolve hemodynamic equations by time dt and update state variables.
        System evolved according to Balloon - Windkessel hemodynamic model. """

        # Hemodynamic response equations
        dx = dt * (x - self._kappa * self._x - self._gamma * (self._f - 1.))
        df = dt * self._x
        dv = (dt / self._tau) * (self._f - self._v ** (1. / self._alpha))
        dq = ((dt * self._f) / (self._tau * self._rho)) * \
             (1. - (1. - self._rho) ** (1. / self._f)) - \
             (dt / self._tau) * self._q * (self._v **((1. / self._alpha) - 1.))

        # Add differences to initial values
        self._x += dx
        self._f += df
        self._v += dv
        self._q += dq

        if self.obata:
            # Obata 2004
            self._y = self._V0 * (self._k1 * (1. - self._q) + self._k2 *
                                  (self._v - self._q) - self._k3 * (1. - self._v))
        else:
            # Friston 2003 (only compatible with parameters in friston03.py)
            self._y = self._V0 * (self._k1*(1.-self._q) +
                self._k2*(1.-self._q/self._v) +
                self._k3*(1.-self._v))

        return

    @property
    def state(self):
        """All hemo state variables, 5 rows by len(nodes) columns.
            Rows are x, v, f, q, y. """
        return np.vstack((self._x, self._f, self._v, self._q, self._y))
