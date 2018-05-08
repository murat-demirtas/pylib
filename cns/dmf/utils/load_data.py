#! /usr/bin/python

""" Functions to load empirical data. """

from os.path import join
from scipy.io import loadmat
from .general import clean_builtins
import numpy as np


def load_model_params():
    """ Returns the model's synaptic parameters defined
    in synaptic.py as a python dictionary. """

    from ..params import synaptic

    params = clean_builtins(vars(synaptic))

    return params

'''
def load_hagmann(file_path):

    human_66 = loadmat(file_path)

    SC = human_66['C']
    FC = human_66['FC_emp']
    n = SC.shape[0]

    # Uncomment the block below for using Human_66.mat
    # # Indexing of rows/columns, Matlab -> Python indexing
    # order = human_66['Order'].astype(int) - 1
    #
    # # Reorder FC
    # for i in range(n):
    #     FC[:, i] = FC[order, i]
    # for i in range(n):
    #     FC[i, :] = FC[i, order]
    #
    # # Reorder SC
    # for i in range(n):
    #     SC[:, i] = SC[order, i]
    # for i in range(n):
    #     SC[i, :] = SC[i, order]

    # Currently unused:
    # human_66['L']
    # human_66['talairach_66']
    # human_66['anat_lbls']

    return SC, FC


def load_hcp(data_dir, hemi="L"):

    # Load data for both hemispheres
    myelin_LR = np.load(join(data_dir, 'myelin.npy'))
    C_L = np.load(join(data_dir, 'sc_L.npy'))
    C_R = np.load(join(data_dir, 'sc_R.npy'))
    empFC_LR = np.load(join(data_dir, 'fc_corr.npy'))
    areas_LR = np.load(join(data_dir, 'area.npy'))

    # Hemisphere specific network membership
    networks = np.loadtxt(join(data_dir, "yeo_17_" + hemi + ".txt"), dtype=int)

    if hemi == 'L':

        # Functional and structural connectivity and parcel areas
        FC = empFC_LR[:50, :50]
        SC = C_L
        areas = areas_LR[2:52]

        # Myelination values
        # NOTE: temporarily clipping outlying myelin value
        myelin = myelin_LR[:50]
        msort = myelin.argsort()
        myelin[msort[-1]] = myelin[msort[-2]]

    elif hemi == 'R':

        # Functional and structural connectivity, parcel areas, myelination
        FC = empFC_LR[50:, 50:]
        SC = C_R
        areas = areas_LR[54:]
        myelin = myelin_LR[50:]

    else:
        raise NotImplementedError("Unrecognized hemisphere argument")

    return SC, FC, areas, myelin, networks
'''