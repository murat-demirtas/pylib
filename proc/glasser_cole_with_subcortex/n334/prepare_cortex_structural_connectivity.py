from utils.io import Data
from utils.dhelper import init_vars
from mri.wrappers import hcp_subjects, hcp_get
from mri.fmri import TimeSeries
from mri.parcels import Parcel
import numpy as np


input_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/raw/'
output_dir = '/Users/md2242/Projects/master/data/glasser_cole_with_subcortex/'

data = Data(input_dir=input_dir, output_dir=output_dir, plot_dir='')
sc_file = data.load('glasser_cole/DWI_WholeBrain.hdf5') # SC matrices
structure_file = data.load('glasser_cole/Structure_MSMAll.hdf5') # for Myelin maps
myelin_file = data.load('glasser_cole/Myelin_BC_MSMAll.hdf5') # for Myelin maps

subjects = hcp_subjects(sc_file)
N_subjects = len(subjects)
variables = init_vars(['sc'], (360, 360, N_subjects))
variables = init_vars(['myelin','thickness'], (360, N_subjects), variables)

## For subjects
for ii, ss in enumerate(subjects):
    # get SC matrix
    variables['sc'][:, :, ii] = hcp_get(sc_file, ss, ['conn_1'])[19:, 19:]
    # get myelin map
    variables['myelin'][:, ii] = hcp_get(myelin_file, ss, ['myelin_map'])
    variables['thickness'][:, ii] = hcp_get(structure_file, ss, ['thickness'])

fname = 'hcp_cortex_structural_final.hdf5'
file_hdf = data.save(fname)
file_hdf.create_dataset('sc', data=variables['sc'],compression="lzf")
file_hdf.create_dataset('myelin', data=variables['myelin'],compression="lzf")
file_hdf.create_dataset('thickness', data=variables['thickness'],compression="lzf")
file_hdf.close()
