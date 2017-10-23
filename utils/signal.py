#! usr/bin/python
import numpy as np
from scipy.signal import gaussian, butter, filtfilt

def get_freq(x, fs):
    freqs = np.fft.fftfreq(x, fs)
    idx = np.array(np.where(freqs >= 0)[0])
    freqs = freqs[idx]
    return freqs, idx


def gaussian_kernel(N, sigma=0.01):
    g = gaussian(N, sigma * N)
    g /= np.sum(g)
    return g


def smooth(x, kernel):
    y = np.empty(x.dim)
    for ii in range(x.shape[0]):
        y[ii] = np.convolve(x[ii], kernel, 'same')
    return y


def bandpass(x, fs, flp=0.04, fhp=0.07, order=2):
    # apply bandpass filter (0.04-0.07 Hz band by default)
    nyq = 0.5 / fs
    Wn = [flp / nyq, fhp / nyq]
    b, a = butter(order, Wn, 'bandpass')
    return filtfilt(b, a, x)#, axis=1)


def powerspectrum(data, N, fs, smooth=True):
    freqs, idx = get_freq(N, fs)
    power = np.abs(np.fft.fft(data)) ** 2
    power = power[idx]
    if smooth:
        kernel = gaussian_kernel(power.shape[0])
        power = np.convolve(power, kernel, 'same')
        #for ii in range(len(self._power)):
        #    self._power[ii] = np.convolve(self._power[ii], self._gauss, 'same')
    return power, freqs


"""
Normalize PSD of each region
"""
def normalize_spectrum(x):
    psd = np.copy(x)
    N = psd.shape[0]
    for ii in range(N):
        psd[ii, :] = psd[ii, :] / psd[ii, :].sum()
    return psd

"""
Calculate band-power
"""
def band_power(x_sim, freqs, bands):
    N = x_sim.shape[0]
    n_bands = len(bands)
    r = np.empty((N, n_bands))
    for ii in range(n_bands):
        i1 = np.abs(freqs - bands[ii][0]).argmin()
        i2 = np.abs(freqs - bands[ii][1]).argmin()
        r[:, ii] = x_sim[:, i1:i2].mean(1)
    return r


def scrub(ts, idx=None):
    for ii in idx:
        ts[:,ii] = (ts[:,ii-1] + ts[:,ii+1])/2
    return ts