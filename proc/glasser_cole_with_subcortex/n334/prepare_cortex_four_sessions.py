from utils.io import Data
from utils.dhelper import init_vars
from mri.wrappers import hcp_subjects, hcp_get
from mri.fmri import TimeSeries
from mri.parcels import Parcel
import numpy as np

input_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/raw/'
output_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/'

data = Data(input_dir=input_dir, output_dir=output_dir, plot_dir='')
fc_file = data.load('BOLD_MSMAll_hp2000_clean.hdf5') # Time-series

subjects = hcp_subjects(fc_file)
N_subjects = len(subjects)
variables = init_vars(['fc_rest_1', 'fc_rest_2', 'fc_rest_3', 'fc_rest_4',
                        'fcov_rest_1', 'fcov_rest_2', 'fcov_rest_3', 'fcov_rest_4'], (360, 360, N_subjects))

## For subjects
for ii, ss in enumerate(subjects):
    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_1']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    variables['fcov_rest_1'][:, :, ii] = ts.cov
    variables['fc_rest_1'][:, :, ii] = ts.fc

    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_2']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    variables['fcov_rest_2'][:, :, ii] = ts.cov
    variables['fc_rest_2'][:, :, ii] = ts.fc

    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_3']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    variables['fcov_rest_3'][:, :, ii] = ts.cov
    variables['fc_rest_3'][:, :, ii] = ts.fc

    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_4']), fs=0.72, standardize=False, demean=True, gsr=False, index=19)
    variables['fcov_rest_4'][:, :, ii] = ts.cov
    variables['fc_rest_4'][:, :, ii] = ts.fc


freqs = ts.freq

fname = 'hcp_cortex_four_sessions_fc.hdf5'
file_hdf = data.save(fname)
g1 = file_hdf.create_group('rest_1')
g1.create_dataset('fc', data=variables['fc_rest_1'],compression="lzf")
g1.create_dataset('fcov', data=variables['fcov_rest_1'],compression="lzf")

g2 = file_hdf.create_group('rest_2')
g2.create_dataset('fc', data=variables['fc_rest_2'],compression="lzf")
g2.create_dataset('fcov', data=variables['fcov_rest_2'],compression="lzf")

g3 = file_hdf.create_group('rest_3')
g3.create_dataset('fc', data=variables['fc_rest_3'],compression="lzf")
g3.create_dataset('fcov', data=variables['fcov_rest_3'],compression="lzf")

g4 = file_hdf.create_group('rest_4')
g4.create_dataset('fc', data=variables['fc_rest_4'],compression="lzf")
g4.create_dataset('fcov', data=variables['fcov_rest_4'],compression="lzf")

file_hdf.close()
