#! usr/bin/python
import numpy as np
from scipy.signal import butter, filtfilt, welch


def get_freq(x, fs):
    """
    Get frequencies and indices
    :param x:
    :param fs:
    :return:
    """
    freqs = np.fft.fftfreq(x, fs)
    idx = np.array(np.where(freqs >= 0)[0])
    freqs = freqs[idx]
    return freqs, idx


def bandpass(x, fs, flp=0.04, fhp=0.07, order=2):
    """
    Bandpass filter
    :param x:
    :param fs:
    :param flp:
    :param fhp:
    :param order:
    :return:
    """
    # apply bandpass filter (0.04-0.07 Hz band by default)
    nyq = 0.5 / fs
    Wn = [flp / nyq, fhp / nyq]
    b, a = butter(order, Wn, 'bandpass')
    return filtfilt(b, a, x)#, axis=1)

def powerspectrum(data, N, fs, smooth=True):
    """
    Power spectral density
    :param data:
    :param N:
    :param fs:
    :param smooth:
    :return:
    """
    #freqs, idx = get_freq(N, fs)
    if data.ndim > 1:
        freqs, power = welch(data, fs=fs, axis=1, nperseg=N/2, noverlap=N/4)
        #power = np.abs(np.fft.fft(data)) ** 2
        #power = power[idx]
        #import pdb; pdb.set_trace()
        #power = power[:,:N/2]
        if smooth:
            kernel = gaussian_kernel(power.shape[0])
            #power = np.convolve(power, kernel, 'same')
            for ii in range(len(power)):
                power[ii] = np.convolve(power[ii], kernel, 'same')
        return power[:,1:], freqs[1:]
    else:
        freqs, power = welch(data, fs=fs, nperseg=N/2, noverlap=N/4)
        power = power[1:]
        if smooth:
            kernel = gaussian_kernel(power.shape[0])
            # power = np.convolve(power, kernel, 'same')
            #for ii in range(len(power)):
            power = np.convolve(power, kernel, 'same')
        return power, freqs[1:]


def band_power(x_sim, freqs, bands):
    """
    Computes band limited power
    :param x_sim:
    :param freqs:
    :param bands:
    :return:
    """
    N = x_sim.shape[0]
    n_bands = len(bands)
    r = np.empty((N, n_bands))
    for ii in range(n_bands):
        i1 = np.abs(freqs - bands[ii][0]).argmin()
        i2 = np.abs(freqs - bands[ii][1]).argmin()
        r[:, ii] = x_sim[:, i1:i2].mean(1)
    return r


def csd(self, Nf = 256):
    """
    Computes cross-spectral density
    :param self:
    :param Nf:
    :return:
    """
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


def gaussian_kernel(N, sigma=0.01):
    """
    Generes a Gaussian kernel
    :param N:
    :param sigma:
    :return:
    """
    g = gaussian(N, sigma * N)
    g /= np.sum(g)
    return g


def smooth(x, kernel):
    """
    Smooth data by a given kernel
    :param x:
    :param kernel:
    :return:
    """
    y = np.empty(x.dim)
    for ii in range(x.shape[0]):
        y[ii] = np.convolve(x[ii], kernel, 'same')
    return y
