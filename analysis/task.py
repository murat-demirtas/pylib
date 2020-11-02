import pandas as pd
import numpy as np
from scipy import signal, special
import statsmodels.api as sm

def linreg(y, x):
    """
    Returns estimated parameters of a linear regression model given y and x
    """
    model = sm.OLS(y, sm.add_constant(x, prepend=False), missing='drop')
    results = model.fit()
    return results.params

def regress_out(y, x):
    """
    Returns residuals of a linear regression model given y and x
    """
    model = sm.OLS(y, sm.add_constant(x, prepend=False), missing='drop')
    results = model.fit()
    return results.resid

def read_ev(path, filename, t_scan, fs=0.72):
    """
        Reads event files in EVs directory. Event files needs to be corrected for timing.

    Keyword arguments:
        Filename (event files), and fs (sampling frequency, by default: 1)

    Returns:
        Binary event vector
    """
    df =pd.read_csv(path +'EVs/' + filename +'.txt', sep='\t', header=None)
    event_time = np.array(df[0].values/fs, dtype=int)
    event_duration = np.array(np.ceil(df[1].values/fs), dtype=int)

    binary_events = np.zeros(t_scan)
    for ii in range(len(event_time)):
        binary_events[event_time[ii]:(event_time[ii ] +event_duration[ii])] = 1.0
    return binary_events

def plot_tasks(ax_in, tasks, task_timings, colors, task_amp=1.0):
    for i in range(len(tasks)):
        ax_in.plot(task_timings[i]*task_amp, lw = 2, color=colors[i])
        label_locs = np.where(task_timings[i]==1)[0]
        xloc = (label_locs[0] + label_locs[1]) / 2.0
        ax_in.text(xloc, task_amp*0.7, tasks[i], fontsize=8, fontweight='bold')

def spm_hrf(dt, a1=6, a2=16, b1=1, b2=1, c=1 / 6, onset=0, A=1, T=32):
    """
    SPM's double gamma HRF with parameters:
        Delay of response (a1) and undershoot (a2)
        Dispersion of response (b1) and undershoot (b2)
        c: Ratio of response/undershoot
        dt: Sampling rate
        T: Length of kernel

    Returns:
        Hemodynamic response function
    """
    t = np.arange(onset, T + onset, dt)
    return A * (t ** (a1 - 1) * b1 ** a1 * np.exp(-b1 * t) / special.gamma(a1) - c * (
                t ** (a2 - 1) * b2 ** a2 * np.exp(-b2 * t) / special.gamma(a2)))

def get_derivatives(hrf_in, fs=1):
    '''
    Calcultes derivatives of a given hemodynamic response function

    Keyword arguments:
        hrf_in: HRF to calculate the derivatives
        fs: Sampling rate
    Returns:
        TD: Temporal derivative
        DD: Dispersion derivative
    '''
    TD = np.diff(hrf_in)
    DD = (hrf_in - spm_hrf(fs, b1=1 + 0.01)) / 0.01
    TD = np.hstack((0, TD))
    DD = np.hstack((0, DD))
    return TD, DD

def gen_desmat_hrf(evs, pscalar, hrf, hrf_td, hrf_dd):
    """
    Generates design matrix

    Keyword arguments:
        evs: list of string containing task events
        pscalar: parcellated scalar file
        hrf: hemodynamic response function
        hrf_td: temporal derivative of hemodynamic response function
        hrf_dd: dispersion derivative of hemodynamic response function

    Returns:
        design matrix
    """
    t_scan = pscalar.shape[0]
    hrf_vects = np.empty((len(evs), t_scan))
    hrf_td_vects = np.empty((len(evs), t_scan))
    hrf_dd_vects = np.empty((len(evs), t_scan))
    bin_vects = np.empty((len(evs), t_scan))
    for ii, lbl in enumerate(evs):
        event_vector_bin = read_ev(lbl, t_scan, fs=0.72)
        bin_vects[ii, :] = np.copy(event_vector_bin)
        hrf_vects[ii, :] = signal.fftconvolve(event_vector_bin, hrf, 'same')
        hrf_td_vects[ii, :] = signal.fftconvolve(event_vector_bin, hrf_td, 'same')
        hrf_dd_vects[ii, :] = signal.fftconvolve(event_vector_bin, hrf_dd, 'same')

    return np.vstack((hrf_vects, bin_vects, hrf_td_vects, hrf_dd_vects))

def gen_fir(event_vector, order=32, lag=1):
    """
    Generates FIR

    Keyword arguments:
        event_vector: Task events as a set of binary vector
        order: Order of FIR
        lag: Lag

    Returns:
        FIR filter
    """
    l = len(event_vector)
    firs = np.zeros((order, l))
    for i in range(order):
        firs[i, lag * i:] = event_vector[:l - lag * i]
    return firs

def gen_desmat_fir(path, tasks, t_scan, order = 10, fs = 0.72):
    """
    Generates design matrix based on FIR approach

    Keyword arguments:
        Paramaters to pass read_ev and gen_fir methods
         path: Path to read EVs
         tasks: EV filenames
         t_scan: Scan time
         order: Order of FIR
         fs: Sampling Frequency

    Returns:
        Design matrix
    """

    design_matrix = np.empty((1, t_scan))
    for ii, lbl in enumerate(tasks):
        event_vector_bin = read_ev(path, lbl, t_scan, fs=0.72)
        firs_temp = gen_fir(event_vector_bin, order=order)
        design_matrix = np.vstack((design_matrix, firs_temp))

    return design_matrix[1:, :]