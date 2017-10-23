import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class Parcel():
    def __init__(self, parc='cole', order=None, subcortex=False):
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.parent_path = os.path.abspath(os.path.join(self.module_path, os.pardir))
        self.template_path = self.parent_path + '/data/parcel_info/'

        self.parcel = pd.read_excel(self.template_path  + parc + '.xlsx')

        if not subcortex: self.parcel[self.parcel.surface=='cortex'].reset_index(drop=True)
        self.n = len(self.parcel)

        self.order = None
        if order is not None:
            self.parcel = self.parcel.loc[order]

        self.indices_L = self.parcel.index[self.parcel.hemi == 'L']
        self.indices_R = self.parcel.index[self.parcel.hemi == 'R']


    def reorder(self, data):
        if data.ndim == 1:
            return data[self.order]
        else:
            if data.shape[0] == data.shape[1]:
                return data[self.order,:][:,self.order]
            else:
                return data[self.order,:]

    def sortby(self, keys):
        self.parcel = self.parcel.sort_values(by=keys)
        self.order = self.parcel.index
        self.parcel = self.parcel.reset_index(drop=True)

    def drop(self, na=True, key=None, value=None):
        if na:
            self.parcel = self.parcel.loc[self.parcel.hemi.dropna().index]
        else:
            if key is not None:
                self.parcel = self.parcel[self.parcel[key] != value]
        if self.order is None:
            self.order = self.parcel.index
        else:
            self.order = self.order[self.parcel.index]
        self.parcel = self.parcel.reset_index(drop=True)
        self.n = len(self.parcel)

    def get_labels(self, hemi=False, dropna=False):
        if dropna:
            self.drop()

        if hemi:
            self.drop()
            label = self.parcel.abbrv.values; h_label = self.parcel.hemi.values
            return [label[ii].encode('utf-8') + '-' + h_label[ii].encode('utf-8') for ii in range(self.n)]
        else:
            return [self.parcel.abbrv.values[ii].encode('utf-8') for ii in range(self.n)]

    def flip_order(self, labels):
        L = len(labels)
        lh_labels = labels[:int(L / 2)]
        rh_labels = labels[int(L / 2):]

        node_order = list()
        node_order.extend(rh_labels)
        node_order.extend(lh_labels[::-1])
        return node_order

    def colorby(self, key, colors=None, palette='Set2'):
        groups = self.parcel.groupby([key])
        group_keys = groups.groups.keys()

        if colors is None:
            data = np.arange(len(group_keys)) + 1
            cmap = plt.get_cmap(palette)
            cNorm = mpl.colors.Normalize(vmin=data[0], vmax=data[-1])
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
            colors = scalarMap.to_rgba(data)

        clusters = np.empty(self.n)
        color_list = np.zeros((self.n, 4))
        for ii, kk in enumerate(group_keys):
            idx = groups.groups[kk]
            color_list[idx,:] = np.tile(colors[ii], (len(idx),1))
            clusters[idx] = ii

        bounds = np.where(np.diff(clusters) != 0.)[0] + 1
        bounds = np.insert(bounds, [0, len(bounds)], [0, self.n])
        group_keys = [group_keys[ii].encode('utf-8') for ii in range(len(group_keys))]
        return color_list, group_keys, bounds, colors

    def network_mask(self, key):
        groups = self.parcel.groupby([key])
        group_keys = groups.groups.keys()

        networks = np.ones((self.n, self.n))
        for kk in group_keys:
            idx = groups.groups[kk]
            for ii in idx:
                for jj in idx:
                    networks[ii,jj] = 0.

        return networks

    """
    Separate left and right hemisphere
    """
    def separate_lr(self, x):
        if x.ndim == 3:
            xl = np.zeros((len(self.indices_L), len(self.indices_R), x.shape[2]))
            xr = np.zeros((len(self.indices_L), len(self.indices_R), x.shape[2]))
            for ii in range(x.shape[2]):
                xdummy = np.copy(x[:,:,ii])
                xl[:, :, ii] =  np.copy(xdummy[self.indices_L, :][:, self.indices_L])
                xr[:, :, ii] = np.copy(xdummy[self.indices_R, :][:, self.indices_R])
        elif x.ndim == 2:
            if x.shape[0] == x.shape[1]:
                xl = x[self.indices_L, :][:, self.indices_L]
                xr = x[self.indices_R, :][:, self.indices_R]
            else:
                xl = x[self.indices_L,:]
                xr = x[self.indices_R,:]
        else:
            xl = x[self.indices_L]
            xr = x[self.indices_R]
        return [xl, xr]


    """
    Concatenate left and right
    """
    def merge_lr(self, x_l, x_r):
        N = len(self.indices_L) + len(self.indices_R)
        if x_l.ndim == 1:
            x_out = np.zeros(N)
            x_out[self.indices_L] = x_l
            x_out[self.indices_R] = x_r
        else:
            x_out = np.empty((N, x_l.shape[1]))
            for ii in range(x_l.shape[1]):
                dummy = np.zeros(N)
                dummy[self.indices_L] = x_l[:, ii]
                dummy[self.indices_R] = x_r[:, ii]
                x_out[:, ii] = np.copy(dummy)
        return x_out

