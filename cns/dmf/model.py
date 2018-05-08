import numpy as np
from core import models as dmf_model

class Dmf():
    def __init__(self, sc, gradient = None, network_mask=None, *args, **kwargs):
        self.sc = sc
        self.gradient = gradient

        if isinstance(self.sc, list):
            if not isinstance(gradient, list):
                self.gradient = [self.gradient, self.gradient]
            if not isinstance(network_mask, list):
                network_mask = [network_mask, network_mask]

            self.dmf = [dmf_model.Model(self.sc[ii], G=1.0, myelin=self.gradient[ii], network_mask=network_mask[ii],
                                        verbose=False, *args, **kwargs) for ii in range(2)]
        else:
            self.dmf = dmf_model.Model(self.sc, G=1.0, myelin=self.gradient, network_mask=network_mask,
                                       verbose=False, *args, **kwargs)

        #print self.dmf._sc_norm


    def set(self, parameter, values, separate = False):
        if isinstance(self.dmf, list):
            if separate:
                [setattr(self.dmf[ii], parameter, values[ii]) for ii in range(2)]
            else:
                [setattr(self.dmf[ii], parameter, values) for ii in range(2)]
        else:
            setattr(self.dmf, parameter, values)


    def get(self, parameter):
        if isinstance(self.dmf, list):
            return [self.dmf[ii].__getattribute__(parameter) for ii in range(2)]
        else:
            return self.dmf.__getattribute__(parameter)


    def check_stability(self, compute_FIC=True):
        if isinstance(self.dmf, list):
            [self.dmf[ii]._update_matrices(compute_FIC=compute_FIC) for ii in range(2)]
            return np.array([self.dmf[ii]._unstable for ii in range(2)]).any()
        else:
            self.dmf._update_matrices(compute_FIC=compute_FIC)
            return self.dmf._unstable


    def moments_method(self, BOLD = True, *args, **kwargs):
        if isinstance(self.dmf, list):
            [self.dmf[ii].moments_method(BOLD = BOLD, *args, **kwargs) for ii in range(2)]
        else:
            self.dmf.moments_method(BOLD = BOLD, *args, **kwargs)


    def psd_syn(self, freqs, pop='E'):
        if isinstance(self.dmf, list):
            csd_1 = self.dmf[0].power_spectrum(freqs, pop=pop)
            csd_2 = self.dmf[1].power_spectrum(freqs, pop=pop)
            psd_E_1 = np.abs(np.array([np.diag(csd_1[:, :, ii]) for ii in xrange(len(freqs))]).T)
            psd_E_2 = np.abs(np.array([np.diag(csd_2[:, :, ii]) for ii in xrange(len(freqs))]).T)
            return [psd_E_1, psd_E_2]
        else:
            csd = self.dmf.power_spectrum(freqs, pop=pop)
            return np.abs(np.array([np.diag(csd[:, :, ii]) for ii in xrange(len(freqs))]).T)


    def csd_syn(self, freqs, pop='E'):
        if isinstance(self.dmf, list):
            csd_1 = self.dmf[0].power_spectrum(freqs, pop=pop)
            csd_2 = self.dmf[1].power_spectrum(freqs, pop=pop)
            #psd_E_1 = np.abs(np.array([np.diag(csd_1[:, :, ii]) for ii in xrange(len(freqs))]).T)
            #psd_E_2 = np.abs(np.array([np.diag(csd_2[:, :, ii]) for ii in xrange(len(freqs))]).T)
            return [csd_1, csd_2]
        else:
            csd = self.dmf.power_spectrum(freqs, pop=pop)
            return csd#np.abs(np.array([np.diag(csd[:, :, ii]) for ii in xrange(len(freqs))]).T)


    def psd_bold(self, freqs):
        if isinstance(self.dmf, list):
            csd_1 = self.dmf[0].power_spectrum_bold(freqs)
            csd_2 = self.dmf[1].power_spectrum_bold(freqs)
            psd_E_1 = np.abs(np.array([np.diag(csd_1[:, :, ii]) for ii in xrange(len(freqs))]).T)
            psd_E_2 = np.abs(np.array([np.diag(csd_2[:, :, ii]) for ii in xrange(len(freqs))]).T)
            return [psd_E_1, psd_E_2]
        else:
            csd = self.dmf.power_spectrum_bold(freqs)
            return np.abs(np.array([np.diag(csd[:, :, ii]) for ii in xrange(len(freqs))]).T)


    def disconnect(self):
        from copy import deepcopy
        self.dmf_disconnect = deepcopy(self.dmf)
        if isinstance(self.dmf_disconnect, list):
            for ii in range(2):
                self.dmf_disconnect[ii]._I_ext = (self.dmf[ii].G * self.dmf[ii]._sc_norm * self.dmf[ii].J_NMDA_EE) \
                                                 * ((self.dmf[ii]._SC.T).dot(self.dmf[ii]._S_E_ss))
                self.dmf_disconnect[ii].G = 0.0
                self.dmf_disconnect[ii].moments_method(BOLD=False)
        else:
            self.dmf_disconnect._I_ext = (self.dmf.G * self.dmf._sc_norm * self.dmf.J_NMDA_EE) \
                                             * ((self.dmf._SC.T).dot(self.dmf._S_E_ss))
            self.dmf_disconnect.G = 0.0
            self.dmf_disconnect.moments_method(BOLD=False)
