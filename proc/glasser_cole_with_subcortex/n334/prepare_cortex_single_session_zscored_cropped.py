from utils.io import Data
from utils.dhelper import init_vars
from mri.wrappers import hcp_subjects, hcp_get
from mri.fmri import TimeSeries
from mri.parcels import Parcel
import numpy as np

def get_tau(autocorrs, lags):
    tau_x = np.empty(autocorrs.shape[1])
    for ii in xrange(autocorrs.shape[1]):
        log_ac = np.log(np.maximum(autocorrs[:,ii], 1e-10))
        lin_reg = np.polyfit(lags, log_ac, 1)
        tau_x[ii] = -1. / lin_reg[0]
    return tau_x

input_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/raw/'
output_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/'

data = Data(input_dir=input_dir, output_dir=output_dir, plot_dir='')
fc_file = data.load('BOLD_MSMAll_hp2000_clean.hdf5') # Time-series

subjects = hcp_subjects(fc_file)
N_subjects = len(subjects)
variables = init_vars(['fc'], (360, 360, N_subjects))

#time_lags = np.arange(0,6)*0.72
#freq_ds = 22

## For subjects
for ii, ss in enumerate(subjects):
    # FC matrix for session 1 and session 2
    # get FC matrix
    ts = TimeSeries(hcp_get(fc_file, ss, ['rest_1', 'rest_2', 'rest_3', 'rest_4']), fs=0.72, standardize=True, demean=False, gsr=False, index=19, crop=100)
    ts._ts = ts._standardize(ts.ts)
    variables['fc'][:,:,ii] = ts.fc

fname = 'hcp_cortex_zscored_single_session_cropped_fc.hdf5'
file_hdf = data.save(fname)
file_hdf.create_dataset('fc', data=variables['fc'],compression="lzf")
file_hdf.close()
