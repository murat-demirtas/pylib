import os
import numpy as np

class Loader:
    def __init__(self, data, functional, structural, parcel):
        functional_file = data.load(functional)
        structural_file = data.load(structural)

        self.thickness = None
        self.myelin = None
        self.fc = None
        self.fc_mat = None
        self.sc = None
        self.indices_L = None
        self.indices_R = None
        self.networks = None

    """
    Load dataset
    """
    def load(self, hierarchy, linearized=False, hemi='LR', session='all', fc_mat=False):
        # Load Functional connectivity
        if session == 'all':
            fin = self.data_input.load('hcp_334_cortex_zscored_single_session.hdf5', from_path=self.data_input.output_dir + 'data/')
            if hemi == 'LR':
                self.fc = fin['fc'].value
                if fc_mat:
                    self.fc_mat = fin['fc_mat'].value
            elif hemi == 'whole':
                self.fc = fin['fc_whole'].value
                if fc_mat:
                    self.fc_mat = fin['fc_mat'].value
            else:
                self.fc = fin[hemi]['fc'].value
                if fc_mat:
                    self.fc_mat = fin[hemi]['fc_mat'].value
            fin.close()
        else:
            fin = self.data_input.load('hcp_334_cortex_zscored_two_sessions.hdf5', from_path=self.data_input.output_dir + 'data/')
            if hemi == 'LR':
                self.fc = fin[session]['fc'].value
                if fc_mat:
                    self.fc_mat = fin[session]['fc_mat'].value
            elif hemi == 'whole':
                self.fc = fin[session]['fc_whole'].value
                if fc_mat:
                    self.fc_mat = fin[session]['fc_mat'].value
            else:
                self.fc = fin[session][hemi]['fc'].value
                if fc_mat:
                    self.fc_mat = fin[session][hemi]['fc_mat'].value
            fin.close()

        # Load Structural Connectivity
        file_structural = self.data_input.load('hcp_334_cortex_average_structural.hdf5', from_path=self.data_input.output_dir + 'data/')
        indices_L = file_structural['indices_L'].value
        indices_R = file_structural['indices_R'].value
        self.indices_L = indices_L
        self.indices_R = indices_R
        sc_raw = file_structural['sc'].value
        #psize = file_structural['psize_mat'].value
        #sc_raw = sc_raw*psize
        file_structural.close()

        #import pdb; pdb.set_trace()

        if hierarchy != 'none':
            #if hierarchy == 'surrogate':
            file_myelin = self.data_input.load('hcp_334_cortex_myelin_maps.hdf5', from_path=self.data_input.output_dir + 'data/')
            measure_raw = file_myelin[hierarchy].value
            file_myelin.close()
        else:
            measure_raw = np.zeros(360)


        if hemi == 'whole':
            if linearized:
                measure_raw = self.linearize_myelin(measure_raw)
            self.sc = self.normalize_sc(sc_raw)
        else:
            measure_l = measure_raw[indices_L]
            measure_r = measure_raw[indices_R]
            if linearized:
                measure_l = self.linearize_myelin(measure_l)
                measure_r = self.linearize_myelin(measure_r)
            sc_l = self.normalize_sc(sc_raw[indices_L, :][:, indices_L])
            sc_r = self.normalize_sc(sc_raw[indices_R, :][:, indices_R])
            if hemi == 'LR':
                measure_raw = [measure_l, measure_r]
                self.sc = [sc_l, sc_r]
            elif hemi == 'L':
                measure_raw = measure_l
                self.sc = sc_l
            else:
                measure_raw = measure_r
                self.sc = sc_r

        if hierarchy != 'none':
            if hierarchy == 'thickness':
                self.thickness = measure_raw
                if hemi == 'LR':
                    self.myelin = [None, None]
            else:
                self.myelin = measure_raw
                if hemi == 'LR':
                    self.thickness = [None, None]
        else:
            self.thickness = None
            self.myelin = None
            if hemi == 'LR':
                self.thickness = [None, None]
                self.myelin = [None, None]

    """
    Load spectral data
    """
    def load_spectral(self, meg=True, bold=False):
        fin = self.data_input.load('hcp_334_cortex_zscored_single_session.hdf5', from_path=self.data_input.output_dir + 'data/')
        if bold:
            self.psd_bold = fin['psd'].value
            self.freqs = fin['freqs'].value

        if meg:
            self.av_psd_meg = fin['av_psd_meg'].value
            self.freqs_meg = fin['freqs_meg'].value

            self.n_bands = 5
            self.bands = [[2., 4.], [4., 8.], [8., 15.], [15., 35.], [35., 50.]]