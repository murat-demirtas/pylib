import numpy as np
from copy import copy
import os
from mri.cifti import Gifti, Cifti


def convert(parcel_in, parcel_out):
    module_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.abspath(os.path.join(module_path, os.pardir))
    template_path = parent_path + '/data/templates/templates_32k/'

    parc_file_in = Cifti(template_path+'cifti/'+parcel_in+'.dlabel.nii')
    scx_offset_in = parc_file_in.stx_offset()
    parc_in = np.squeeze(parc_file_in.data)

    parc_file_out = Cifti(template_path + 'cifti/'+parcel_out+'.dlabel.nii')
    scx_offset_out = parc_file_out.stx_offset()
    parc_out = np.squeeze(parc_file_out.data)

    unique_parc_in = np.unique(parc_in[:scx_offset_in])
    unique_parc_out = np.unique(parc_out[:scx_offset_out])

    maxoverlap = np.zeros(len(unique_parc_in))
    index = np.zeros(len(unique_parc_in))
    for ii, pp in enumerate(unique_parc_in):
        dummy2 = parc_out[parc_in == pp]

        count = np.zeros(len(np.unique(dummy2)))
        for jj, pp2 in enumerate(np.unique(dummy2)):
            count[jj] = (dummy2 == pp2).sum()

        maxoverlap[ii] = count.max() / count.sum()
        index[ii] = int(np.where(unique_parc_out == np.unique(dummy2)[count.argmax()])[0][0])

    if maxoverlap.mean() < 1:
        print "Warning: Parcel labels do not overlap. Check dlabel file."

    return index.astype(int)

