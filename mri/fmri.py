#! usr/bin/python

import numpy as np
from scipy.signal import gaussian, butter, filtfilt, hilbert
from math import pi
from scipy.stats.mstats import zscore
from utils.stats import xcorr, xcov
from matplotlib import mlab
import statsmodels.api as sm

class TimeSeries(object):
    def __init__(self, ts_raw, fs=2, demean=False, standardize=False, gsr=False, index=None, crop=0):
        # Provide time series columns as regions, rows as time (N x T)
        # fs is the sampling frequency (by default 2)
        ##########################################################################
        if isinstance(ts_raw, list):
            if index is None:
                ts_new = np.empty((ts_raw[0].shape[0],1))
            else:
                ts_new = np.empty((ts_raw[0][index:,:].shape[0], 1))

            for ts_list in range(len(ts_raw)):
                if demean:
                    ts_dummy = self._demean(ts_raw[ts_list][index:,crop:])
                elif standardize:
                    ts_dummy = self._standardize(ts_raw[ts_list][index:,crop:])
                    #ts_dummy = self._isscrub(ts_dummy, crop)
                else:
                    ts_dummy = ts_raw[ts_list][index:,crop:]
                ts_new = np.hstack((ts_new, ts_dummy))
            ts_raw = ts_new[:,1:]
        else:
            if demean:
                ts_raw = self._demean(ts_raw[index:,crop:])
            elif standardize:
                ts_raw = self._standardize(ts_raw[index:,crop:])
            else:
                ts_raw = ts_raw[index:,:]

        # Define Spatial Dimensions
        self._N = ts_raw.shape[0]
        self._T = ts_raw.shape[1]
        self._NC = self._N * (self._N - 1) / 2
        # Diagonal indices
        self._ldi = np.tril_indices(self._N, k=-1) # lower diagonal
        self._udi = np.triu_indices(self._N, k=1) # upper diagonal
        self._isd = self._udi # by default, use lower diagonal
        # Initialize raw time series (demeaned or standardized)
        self._fs = fs
        self._ts_raw = ts_raw

        self.reset()
        self._cropped_time = 10

        if gsr: self.gsr()

    #############################################################################
    # Private functions
    #############################################################################
    def _init_hilbert(self):
        self._tcrop = np.arange(self._cropped_time, self._T - self._cropped_time)
        self._ts_full = self._ts
        self._ts = self._ts[:, self._tcrop]
        self._T = self._ts.shape[1]
        self._hilbert_transformed = True
        # Initialize hilbert transform and plv parameters
        self._H = np.empty((self._N, self._T), dtype=complex)
        self._phase = np.empty((self._N, self._T))
        self._amplitude = np.empty((self._N, self._T))
        self._delta_phase = np.empty((self._NC, self._T))
        self._plv = np.empty((self._NC, self._T))
        self._kuramoto = np.empty(self._T)


    def _isscrub(self, ts_in, crop_ind):
        gs = ts_in.mean(0)
        if (gs >= 5).any():
            ts_in = ts_in[:, np.where(gs < 5)[0]]
            n_scrub = (gs>5).sum()
            ts_in = ts_in[:,(crop_ind-n_scrub):]
        else:
            ts_in = ts_in[:, crop_ind:]
        ts_in = self._standardize(ts_in)

        return ts_in

    def _demean(self, ts):
        return ts - np.transpose(np.tile(np.mean(ts, axis=1), [ts.shape[1], 1]))
        #self._ts_raw = self._ts_raw - np.transpose(np.tile(np.mean(self._ts_raw,axis=1),[self._T,1]))

    def _standardize(self, ts):
        return zscore(ts, axis=1)
        #mu = np.transpose(np.tile(np.mean(self._ts_raw, axis=1), [self._T, 1]))
        #sigma = np.transpose(np.tile(np.std(self._ts_raw, axis=1), [self._T, 1]))
        #self._ts_raw = (self._ts_raw - mu)/sigma

    def _get_freq(self):
        self._ffts = np.fft.fft(self._ts_raw)
        self._freqs = np.fft.fftfreq(self._T, self._fs)
        self._idx = np.argsort(self._freqs)
        # TODO: fix frequency index!! And optimize
        if self._idx.shape[0] % 2 == 0:
            self._idx = self._idx[(int(np.max(self._idx) / 2) + 1):]
        else:
            self._idx = self._idx[int(np.max(self._idx) / 2):]
        self._freqs = self._freqs[self._idx]
        self._gauss = self._gaussian_kernel(len(self._freqs))

    def _gaussian_kernel(self, N, sigma = 0.01):
        g = gaussian(N, sigma * N)
        g /= np.sum(g)
        return g

    #############################################################################
    # Main functions
    #############################################################################
    def gsr(self):
        x = self._ts.mean(0)
        for ii in range(self._N):
            self._ts[ii,:] = sm.OLS(self._ts[ii,:], sm.add_constant(x, prepend=False), missing='drop').fit().resid

    def reset(self):
        # resets the class (set ts as raw ts)
        self._ts = self._ts_raw
        self._power = None
        self._idx = None
        self._hilbert_transformed = False

    def bandpass(self, flp=0.04, fhp=0.07, order=2):
        # apply bandpass filter (0.04-0.07 Hz band by default)
        nyq = 0.5 / self._fs
        Wn = [flp/nyq, fhp/nyq]
        b, a = butter(order, Wn, 'bandpass')
        self._ts = filtfilt(b, a, self._ts, axis=1)

    def hilbert(self):
        # apply hilbert transform; compute phase, amplitude and kuromoto order parameter
        if not self._hilbert_transformed: self._init_hilbert()
        self._H = hilbert(self._ts_full, axis=1)[:, self._tcrop]
        self._phase = np.angle(self._H)
        self._amplitude = np.abs(self._H)
        self._kuramoto = np.abs(np.sum((np.cos(self._phase) + 1j * np.sin(self._phase)) / self._N, axis=0))

    def set_crop(self, crop):
        # set time bins to crop beginning and the end (necessary for hilbert transform)
        self._cropped_time = crop

    def phase_lock(self, method=None):
        # compute phase lock values based on hilbert transform
        # method defines the normalization of phase differences (manual or cosine)
        for t in range(self._T):
            phase_matrix = np.tile(self._phase[:, t], (self._N, 1))
            self._delta_phase[:,t] = phase_matrix[self._isd] - phase_matrix.transpose()[self._isd]
        if method == None:

            self._plv = abs(self._delta_phase)
            self._plv[np.where(self._plv > pi)] = 2 * pi - self._plv[np.where(self._plv > pi)]
            self._plv = 1 - self._plv / pi
        else:
            self._plv = np.cos(self._delta_phase)
    def sliding_window(self, window = 10, step = 5):
        # compute dfc using sliding window analysis
        self._slide = np.arange(0, self._T - window, step)
        self._dfcs = np.zeros((self._NC, len(self._slide)))
        for ii, t in enumerate(self._slide):
            self._dfcs[:, ii] = np.corrcoef(self._ts[:, t:(t + window)])[self._isd]

    def powerspectrum(self, smooth=True):
        # compute power spectrum
        if self._idx == None: self._get_freq()
        self._power = np.abs(np.fft.fft(self._ts)) ** 2
        self._power = self._power[:, self._idx]
        if smooth:
            for ii in range(len(self._power)):
                self._power[ii] = np.convolve(self._power[ii], self._gauss, 'same')

    def csd(self, Nf = 256):
        cs_density = np.zeros((self._N, self._N, Nf + 1), dtype=complex)
        self.freqs_csd = np.linspace(0.0, 0.5/self._fs, Nf + 1)
        for ii in range(self._N):
            for jj in range(ii, self._N):
                cs_density[ii, jj, :] = mlab.csd(self.ts[ii, :], self.ts[jj, :], NFFT=2*Nf, Fs=1. / self._fs)[0]
                #self.cs_density[jj, ii, :] = self.cs_density[ii, jj, :]

        self.cs_density = np.zeros((self._NC, Nf + 1), dtype=complex)
        self.cs_power = np.zeros((self._N, Nf + 1), dtype=complex)
        for ii in range(Nf + 1):
            self.cs_density[:,ii] = cs_density[:,:,ii][self._isd]
            self.cs_power[:,ii] = np.diag(cs_density[:,:,ii])


    def fc_tau(self, tau):
        return xcorr(self._ts, lag=tau)

    def cov_tau(self, tau):
        return xcov(self._ts, lag=tau)

    def auto_corr(self, tau):
        ac = np.empty(self._N)
        for ii in xrange(self._N):
            ac[ii] = xcorr(self._ts[ii,:], lag=tau)
        return ac


    #############################################################################
    # Properties
    #############################################################################

    # Raw timeseries
    @property
    def ts_raw(self):
        return self._ts_raw

    @property
    def mean_raw(self):
        return np.mean(self._ts_raw, axis=1)

    @property
    def var_raw(self):
        return np.var(self._ts_raw, axis=1)

    @property
    def fc_raw(self):
        return np.corrcoef(self._ts_raw)

    @property
    def fc_raw_diag(self):
        return np.corrcoef(self._ts_raw)[self._isd]

    @property
    def cov_raw(self):
        return np.cov(self._ts_raw)

    # Processed timeseries
    @property
    def ts(self):
        return self._ts

    @property
    def mean(self):
        return np.mean(self._ts, axis=1)

    @property
    def var(self):
        return np.var(self._ts, axis=1)

    @property
    def fc(self):
        return np.corrcoef(self._ts)

    @property
    def fc_diag(self):
        return np.corrcoef(self._ts)[self._isd]

    @property
    def cov(self):
        return np.cov(self._ts)

    @property
    def cov_diag(self):
        return np.cov(self._ts)[self._isd]

    @property
    def gas(self):
        return self._ts.mean(0)

    # Time information
    @property
    def time_reduced(self):
        return self._tcrop

    @property
    def time_slide(self):
        return self._slide

    # Spectral information
    @property
    def spect_pow(self):
        return self._power

    @property
    def max_pow(self):
        return self._power.max(axis=1)

    @property
    def max_pow_freq(self):
        return self._freqs[self._power.argmax(axis=1)]

    @property
    def freq(self):
        return self._freqs

    # Dynamic functional connectivity
    @property
    def dfc(self):
        return self._dfcs

    @property
    def dfc_av(self):
        return np.mean(self._dfcs,axis=1)

    @property
    def dfc_var(self):
        return np.var(self._dfcs,axis=1)

    @property
    def plv(self):
        return self._plv

    @property
    def plv_av(self):
        return np.mean(self._plv,axis=1)

    @property
    def plv_var(self):
        return np.var(self._plv,axis=1)

    @property
    def env_fc(self):
        return np.corrcoef(self._amplitude)[self._isd]

    # Hilbert transform
    @property
    def phase(self):
        return self._phase

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def kop(self):
        return self._kuramoto

    @property
    def coherence(self):
        return np.mean(self._kuramoto)

    @property
    def metastability(self):
        return np.std(self._kuramoto)

    @property
    def delta_phase(self):
        return self._delta_phase

    @property
    def gas_env(self):
        return np.abs(hilbert(self._ts.mean(0)))


    '''
    def surrogates(x):
        from math import pi
        N = x.shape[0]
        M = x.shape[1]
        s = np.fft.fft(x)
        phase_rnd = np.zeros(N)
        if N%2 == 1:
            ph = 2*pi*numpy.random.random((1,(N-1)/2)) - pi
            phase_rnd[2:N] = [ph, -ph[::-1]]
        else:
            ph[1:(N - 2) / 2] = 2 * pi * numpy.random.random((1, (N - 2) / 2)) - pi
            phase_rnd[2:N] = [ph, -ph[::-1]]

        s_rnd = np.zeros((N,M))
        for ii in range(M):
            #s_rnd[:,ii] = abs(s[:,ii])*np.exp()

        # inverse fft

        # output
    '''