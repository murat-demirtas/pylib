from tools.io import Data
from tools import utils
import numpy as np

subjects = ['ta5679',
            'ta5880', 'ta5927','ta6143','ta6155','ta6313',
            'ta6325', 'ta6508','ta6665','ta6698','ta6764',
            'ta7061', 'ta7099','ta7244','ta7396','ta7496',
            'ta8113', 'ta8182','ta8215','ta8468','ta8787',
            'ta9168', 'ta9478','ta9588']


from tools.cifti import Cifti
from tools import linalg
fc_control = np.zeros((360,360,24))
fc_ketamine = np.zeros((360,360,24))
fc_agg = np.zeros((64620, 48))
idx = 0
for ii, ss in enumerate(subjects):
    cfile = Cifti('/Users/md2242/Projects/lib/data/dp5/'+ss+'-boldfixica1_8_res-VWMWB_glasser.ptseries.nii')
    scrub = np.loadtxt('/Users/md2242/Projects/lib/data/dp5/'+ss+'_boldfixica1_8.use')
    ts = np.array(cfile.data.squeeze().T)
    ts = ts[:,scrub==1.]
    ts = ts[19:,:]
    fc_control[:,:,ii] = np.corrcoef(ts)
    fc_agg[:,ii] = linalg.subdiag(fc_control[:,:,ii])

    cfile = Cifti('/Users/md2242/Projects/lib/data/dp5/'+ss+'-boldfixica10_17_res-VWMWB_glasser.ptseries.nii')
    scrub = np.loadtxt('/Users/md2242/Projects/lib/data/dp5/'+ss+'_boldfixica10_17.use')
    ts = np.array(cfile.data.squeeze().T)
    ts = ts[:,scrub==1.]
    ts = ts[19:, :]
    fc_ketamine[:, :, ii] = np.corrcoef(ts)
    fc_agg[:,ii+24] = linalg.subdiag(fc_ketamine[:, :, ii])

data = Data(output_dir='/Users/md2242/Projects/lib/data/glasser_cole/processed/', plot_dir='')
fname = 'ketamine_24_connectivity_cortex.hdf5'
file_hdf = data.save(fname)
file_hdf.create_dataset('fc_control', data=fc_control,compression="lzf")
file_hdf.create_dataset('fc_ketamine', data=fc_ketamine,compression="lzf")
file_hdf.create_dataset('fc_agg', data=fc_agg,compression="lzf")

file_hdf.close()
