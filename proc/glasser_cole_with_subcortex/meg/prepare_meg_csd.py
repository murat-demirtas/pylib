import numpy as np
import pandas as pd
from tools.parcels import Parcel
from tools.io import Data

glasser = Parcel()
glasser.drop(na=False, key='surface', value='subcortex')
#indices = glasser.parcel.index[glasser.parcel.hemi == 'L']
indices = glasser.parcel.index

N = indices.shape[0]
data = Data(output_dir='/Users/md2242/Projects/lib/data/glasser_cole/processed/', plot_dir='')

meg_csd_file = pd.read_hdf(data.output_dir + '/meanPSDs.h5')
#meg_csd_file = pd.read_hdf('glasser_cole/processed/csd_subjectAvg_normalized.h5')

meg_freqs = meg_csd_file['mrksSAM_deltaToLowGamma']['default'][101].index.values#[101][101].index.values
Nf = len(meg_freqs)

meg_psd = np.zeros((N, Nf), dtype=complex)
parcel_scalar = glasser.parcel.scalar[indices].values

for ii1, ss1 in enumerate(parcel_scalar):
    if (meg_csd_file['mrksSAM_deltaToLowGamma']['default'].keys() == ss1).any():
        dummy = np.copy(meg_csd_file['mrksSAM_deltaToLowGamma']['default'][ss1].values)
        meg_psd[ii1, :] = dummy

meg_psd = np.abs(meg_psd)
meg_psd_norm = np.zeros(meg_psd.shape)
for ii in range(N): meg_psd_norm[ii,:] = meg_psd[ii,:] / meg_psd[ii,:].sum()

fname = 'hcp_meg_csd.hdf5'
file_hdf = data.save(fname)
file_hdf.create_dataset('freqs', data=meg_freqs,compression="lzf")
file_hdf.create_dataset('psd', data=meg_psd_norm,compression="lzf")
file_hdf.create_dataset('psd_raw', data=meg_psd,compression="lzf")
file_hdf.close()
