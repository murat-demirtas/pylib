from utils.cifti import Cifti
import os
import numpy as np
import pandas as pd
from os import system

def read_table(filename):
    table_raw = pd.read_csv(filename, header=None)
    rois = table_raw[::2].reset_index()
    values = table_raw[1::2]

    #scalar = np.zeros(len(values))
    #for ii in range(len(values)):
    #    scalar[ii] = int(values.iloc[ii][0].split()[0])
    #scalar = pd.DataFrame(scalar)

    rois = rois[0].values
    return rois


def gen_parcel_info(dlabel_file, map, dlabel_path = '', output_path=None):
    module_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = module_dir + '/tmp/'

    dlabel_fname = dlabel_path + dlabel_file + '.dlabel.nii'
    dlabel_cifti = Cifti(dlabel_fname)
    np.squeeze(dlabel_cifti.data)

    system('wb_command -cifti-label-export-table ' +
              dlabel_fname + ' ' + map + ' ' +
              temp_dir + 'keys.txt')

    rois = read_table(temp_dir + 'keys.txt')
    system('rm ' + temp_dir + 'keys.txt')

    label_info = dlabel_cifti.get_labels()
    dlabels = np.squeeze(dlabel_cifti.data)

    n_rois = len(rois)
    hemi_label = ['']*n_rois
    surface_label = [''] * n_rois
    psize = np.zeros(n_rois)
    scalars = np.zeros(n_rois)
    for ii, ll in enumerate(np.unique(dlabels)):
        if (label_info['L']['labels'] == ll).any():
            hemi_label[ii] = 'L'
            surface_label[ii] = 'cortex'
        elif (label_info['R']['labels'] == ll).any():
            hemi_label[ii] = 'R'
            surface_label[ii] = 'cortex'
        else:
            hemi_label[ii] = 'NA'
            surface_label[ii] = 'subcortex'
        psize[ii] = (dlabels == ll).sum()
        scalars[ii] = int(ll)


    parcels = pd.DataFrame({'scalar': scalars,
                            'label': rois,
                            'hemi': hemi_label,
                            'surface': surface_label,
                            'parcel_size': psize})

    if output_path is None:
        parent_path = os.path.abspath(os.path.join(module_dir, os.pardir))
        parcels.to_excel(parent_path + '/data/parcel_info/' + dlabel_file + '.xlsx')
    else:
        parcels.to_excel(output_path + dlabel_file + '.xlsx')