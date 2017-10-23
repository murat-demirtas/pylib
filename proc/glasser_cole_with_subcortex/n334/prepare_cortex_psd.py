from utils.io import Data
from utils.dhelper import init_vars
from mri.wrappers import hcp_subjects, hcp_get
from mri.fmri import TimeSeries
from mri.parcels import Parcel
import numpy as np

input_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/raw/'
output_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/'

data = Data(input_dir=input_dir, output_dir=output_dir, plot_dir='')
fc_file = data.load('glasser_cole/BOLD_MSMAll_hp2000_clean.hdf5') # Time-series

subjects = hcp_subjects(fc_file)
N_subjects = len(subjects)
variables = init_vars(['psd_s1', 'psd_s2', 'psd_s3', 'psd_s4'], (360, 600, N_subjects))

## For subjects
for ii, ss in enumerate(subjects):
    # get SC matrix
    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_1']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    ts.powerspectrum()
    variables['psd_s1'][:, :, ii] = ts.spect_pow
    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_2']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    ts.powerspectrum()
    variables['psd_s2'][:, :, ii] = ts.spect_pow
    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_3']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    ts.powerspectrum()
    variables['psd_s3'][:, :, ii] = ts.spect_pow
    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_4']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    ts.powerspectrum()
    variables['psd_s4'][:, :, ii] = ts.spect_pow

freqs = ts.freq
fname = 'hcp_cortex_four_sessions_psd.hdf5'
file_hdf = data.save(fname)
file_hdf.create_dataset('freqs', data=freqs,compression="lzf")
file_hdf.create_dataset('psd_s1', data=variables['psd_s1'],compression="lzf")
file_hdf.create_dataset('psd_s2', data=variables['psd_s2'],compression="lzf")
file_hdf.create_dataset('psd_s3', data=variables['psd_s3'],compression="lzf")
file_hdf.create_dataset('psd_s4', data=variables['psd_s4'],compression="lzf")
file_hdf.close()
