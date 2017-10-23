import numpy as np
from scipy.stats import spearmanr
from utils.transform import subdiag
from utils.parcels import Parcel
from wrappers import concat_lr

class Network:
    def __init__(self, parcel, networks, sensory, association, subcortex=False):
        parcel_obj = Parcel()
        if not subcortex:
            parcel_obj.drop(na=False, key='surface', value='subcortex')
        self.parcel = parcel_obj.parcel
        self.n_roi = len(parcel_obj.parcel)

        self.indices_L = parcel_obj.indices_L
        self.indices_R = parcel_obj.indices_R

        if networks is None:
            self.networks = ['Auditory', 'Visual', 'Somatomotor',
                             'Dorsal-attention', 'Frontoparietal', 'Ventral-attention',
                             'Default-Mode', 'Cingulo-Opercular']
        else:
            self.networks = networks

        self.parcel_L = self.parcel.loc[self.indices_L].reset_index(drop=True)
        self.parcel_R = self.parcel.loc[self.indices_R].reset_index(drop=True)

        if sensory is None:
            self.sensory_networks = ['Visual', 'Auditory', 'Somatomotor']
        else:
            self.sensory_networks = sensory

        if association is None:
            self.association_networks = ['Ventral-attention', 'Frontoparietal', 'Dorsal-attention',
                                         'Cingulo-Opercular', 'Default-Mode']
        else:
            self.association_networks = association

        for ii, key in enumerate(self.sensory_networks):
            if ii == 0:
                self.sensory = self.parcel.index[self.parcel['network'] == key]
            else:
                self.sensory = self.sensory.append(self.parcel.index[self.parcel['network'] == key])

        for ii, key in enumerate(self.association_networks):
            if ii == 0:
                self.association = self.parcel.index[self.parcel['network'] == key]
            else:
                self.association = self.association.append(
                    self.parcel.index[self.parcel['network'] == key])


    """
    Get networks matrix
    """
    def get_network_matrix(self, x, keys=None, hemi='LR'):
        if keys is None: keys = self.networks
        if hemi == 'LR':
            parcel = self.parcel
        elif hemi == 'L':
            parcel = self.parcel_L
        else:
            parcel = self.parcel_R

        out = np.zeros((len(keys),len(keys)))
        for ii1, key1 in enumerate(keys):
            for ii2, key2 in enumerate(keys):
                if ii1 == ii2:
                    out[ii1, ii2] = subdiag(x[parcel.index[parcel['network'] == key1], :][:,parcel.index[parcel['network'] == key2]]).mean()
                else:
                    out[ii1, ii2] = x[parcel.index[parcel['network'] == key1], :][:,parcel.index[parcel['network'] == key2]].mean()

        return out


    """
    Network average
    """
    def get_networks(self, x, keys=None, hemi='LR', std=False):
        if keys is None: keys = self.networks
        if hemi == 'LR':
            parcel = self.parcel
        elif hemi == 'L':
            parcel = self.parcel_L
        else:
            parcel = self.parcel_R

        if x.ndim == 1:
            out = np.zeros(len(keys))
            if std:
                std_out = np.zeros(len(keys))
            for ii, key in enumerate(keys):
                out[ii] = x[parcel.index[parcel['network'] == key]].mean()
                if std:
                    std_out[ii] = x[parcel.index[parcel['network'] == key]].std()
        else:
            out = np.zeros((len(keys), x.shape[1]))
            if std:
                std_out = np.zeros((len(keys), x.shape[1]))
            for ii, key in enumerate(keys):
                out[ii,:] = x[parcel.index[parcel['network'] == key],:].mean(0)
                if std:
                    std_out[ii, :] = x[parcel.index[parcel['network'] == key], :].std(0)

        if std:
            return out, std_out
        else:
            return out


    def gen_system_idx(self, parcel):
        for ii, key in enumerate(self.sensory_networks):
            if ii == 0:
                sensory = parcel.index[parcel['network'] == key]
            else:
                sensory = sensory.append(parcel.index[parcel['network'] == key])

        for ii, key in enumerate(self.association_networks):
            if ii == 0:
                association = parcel.index[parcel['network'] == key]
            else:
                association = association.append(parcel.index[parcel['network'] == key])

        return sensory, association


    """
    Network average
    """
    def get_systems(self, x, hemi='LR', std=False):
        if hemi == 'L':
            sensory, association = self.gen_system_idx(self.parcel_L)
        elif hemi == 'R':
            sensory, association = self.gen_system_idx(self.parcel_R)
        else:
            sensory, association = self.gen_system_idx(self.parcel)


        if x.ndim == 1:
            out = np.zeros(2)
            out[0] = x[sensory].mean()
            out[1] = x[association].mean()
            if std:
                std_out = np.zeros(2)
                std_out[0] = x[sensory].std()
                std_out[1] = x[association].std()
        else:
            out = np.zeros((2, x.shape[1]))
            out[0,:] = x[sensory,:].mean(0)
            out[1, :] = x[association, :].mean(0)
            if std:
                std_out = np.zeros((2, x.shape[1]))
                std_out[0, :] = x[sensory, :].std(0)
                std_out[1, :] = x[association, :].std(0)

        if std:
            return out, std_out
        else:
            return out

    """
    Create network mask
    """
    def network_mask(self, x, hemi='LR'):
        out = np.zeros(self.n_roi)
        if hemi == 'LR':
            for ii, key in enumerate(self.networks):
                out[self.parcel.index[self.parcel['network'] == key]] = x[ii]
        elif hemi == 'L':
            for ii, key in enumerate(self.networks):
                out[self.parcel_L.index[self.parcel_L.parcel['network'] == key]] = x[ii]
        else:
            for ii, key in enumerate(self.networks):
                out[self.parcel_R.index[self.parcel_R.parcel['network'] == key]] = x[ii]
        return out


    """
    Compare networks
    """
    def compare_networks(self, x, y, keys=None, hemi='LR', corr='spearman'):
        if keys is None: keys = self.networks
        if hemi == 'LR':
            parcel = self.parcel
        elif hemi == 'L':
            parcel = self.parcel_L
        else:
            parcel = self.parcel_R

        if x.ndim == 1:
            out = np.zeros(len(keys))
            for ii, key in enumerate(keys):
                if corr == 'spearman':
                    out[ii] = spearmanr(x[parcel.index[parcel['network'] == key]], y[parcel.index[parcel['network'] == key]])[0]
                else:
                    out[ii] = np.corrcoef(x[parcel.index[parcel['network'] == key]], y[parcel.index[parcel['network'] == key]])[0,1]
        else:
            out = np.zeros((len(keys), x.shape[1]))
            for jj in range(x.shape[1]):
                for ii, key in enumerate(keys):
                    if corr == 'spearman':
                        out[ii,jj] = spearmanr(x[parcel.index[parcel['network'] == key],jj], y[parcel.index[parcel['network'] == key],jj])[0]
                    else:
                        out[ii,jj] = np.corrcoef(x[parcel.index[parcel['network'] == key],jj], y[parcel.index[parcel['network'] == key]],jj)[0, 1]
        return out


    """
    Within network fit
    """
    def within_network_fit(self, data_model, data_emp, hemi='LR'):
        if hemi == 'LR':
            networks = self.networks
            parcel_L = self.parcel_L
            parcel_R = self.parcel_R
            within_net = np.zeros(len(networks))
            for ii, key in enumerate(networks):
                ix_l = parcel_L.index[parcel_L['network'] == key]
                ix_r = parcel_R.index[parcel_R['network'] == key]
                within_net[ii] = \
                            np.corrcoef(concat_lr([data_model[0][ix_l, :][:, ix_l], data_model[1][ix_r, :][:, ix_r]]),
                            concat_lr([data_emp[0][ix_l, :][:, ix_l], data_emp[1][ix_r, :][:, ix_r]]))[0, 1]
        else:
            networks = self.networks
            if hemi == 'L':
                parcel = self.parcel_L
            elif hemi == 'R':
                parcel = self.parcel_R
            else:
                parcel = self.parcel

            within_net = np.zeros(len(networks))
            for ii, key in enumerate(networks):
                ix = parcel.index[parcel['network'] == key]
                within_net[ii] = np.corrcoef(subdiag(data_model[0][ix, :][:, ix]), subdiag(data_emp[0][ix, :][:, ix]))[0, 1]
        return within_net


    """
    Average whole-brain fit across networks
    """
    def whole_brain_network_fit(self, x, x_emp):
        from copy import copy
        data = [np.copy(x[0]), np.copy(x[1])]
        data_emp = [np.copy(x_emp[0]), np.copy(x_emp[1])]

        networks = self.networks
        parcel_L = self.parcel_L
        parcel_R = self.parcel_R
        N_regions = data_emp[0].shape[0]
        tril_ind = np.tril_indices(N_regions, -1)
        for ii in xrange(N_regions):
            data[0][ii, ii] = 0.0
            data[1][ii, ii] = 0.0
            data_emp[0][ii, ii] = 0.0
            data_emp[1][ii, ii] = 0.0

        data[0][tril_ind] = 0.0
        data[1][tril_ind] = 0.0
        data_emp[0][tril_ind] = 0.0
        data_emp[1][tril_ind] = 0.0

        whole_fit = np.zeros(len(networks))
        for ii, key in enumerate(networks):
            ix_l = parcel_L.index[parcel_L['network'] == key]
            ix_r = parcel_R.index[parcel_R['network'] == key]
            dummy_l = np.copy(data[0][ix_l, :])
            dummy_r = np.copy(data[1][ix_r, :])
            dummy_l_emp = np.copy(data_emp[0][ix_l, :])
            dummy_r_emp = np.copy(data_emp[1][ix_r, :])

            dummy = np.hstack((dummy_l[dummy_l != 0], dummy_r[dummy_r != 0]))
            dummy_emp = np.hstack((dummy_l_emp[dummy_l_emp != 0], dummy_r_emp[dummy_r_emp != 0]))

            whole_fit[ii] = np.corrcoef(dummy, dummy_emp)[0, 1]
        return whole_fit

