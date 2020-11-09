import nibabel as nib
import numpy as np
import nibabel.gifti as gifti
import os
from scipy import stats

def get_networks(parcel_file = 'Glasser_NetworkPartition_v9.dlabel.nii', n_parcels = 360):
    module_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.abspath(os.path.join(module_path, os.pardir))
    template_dir = parent_path + '/data/templates/templates_32k/cifti/'
    
    network_labels = Cifti(template_dir + parcel_file)
    networks = [network_labels._header.get_axis(0).label[0][i][0][:3] for i in range(1, n_parcels + 1)]
    network_names = np.unique(networks)

    network_indices = {}
    for key in network_names:
        network_indices[key] = np.where(np.array([networks[i] == key for i in range(n_parcels)]))

    return network_names, network_indices

class Cifti():
    def __init__(self, fname, data = None):
        if data is not None:
            self.input_dir = data.input_dir
            self.output_dir = data.output_dir
        else:
            self.input_dir = ''
            self.output_dir = ''

        of = nib.load(self.input_dir + fname)
        self._header = of.header
        self._data = of.get_fdata()

    def get_structures():
        brain_models = self._header.get_axis(1)
        
    def get_labels(self):
        dlabels = self.data
        label_info = {}

        n_vertices = self._header.get_axis(1).nvertices['CIFTI_STRUCTURE_CORTEX_LEFT']
        
        left_surf, right_surf = np.zeros(n_vertices), np.zeros(n_vertices)
        for name, data_indices, model in self._header.get_axis(1).iter_structures():
            if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
                vertex_indices = model.vertex
                left_scalars = dlabels.T[data_indices]
                left_labels = np.unique(left_scalars)
                left_surf[vertex_indices] = left_scalars
                
                left_scalars_array = np.zeros(n_vertices)
                for ii, ss in enumerate(left_labels):
                    left_scalars_array[left_surf == ss] = ii + 1

                label_info['L'] = {}

                label_info['L']['labels'] = left_labels
                label_info['L']['scalars'] = left_surf
                label_info['L']['scalars_array'] = left_scalars_array
            if name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                vertex_indices = model.vertex
                right_scalars = dlabels.T[data_indices]
                right_labels = np.unique(right_scalars)
                right_surf[vertex_indices] = right_scalars
                
                right_scalars_array = np.zeros(n_vertices)
                for ii, ss in enumerate(right_labels):
                    right_scalars_array[right_surf == ss] = ii + 1

                label_info['R'] = {}

                label_info['R']['labels'] = right_labels
                label_info['R']['scalars'] = right_surf
                label_info['R']['scalars_array'] = right_scalars_array
                
        return label_info

    def parcellate(self, dlabel_fname, zscore=True):
        """
        Parcellates a given dlabel scalar values.

        Keyword arguments:
            dlabel: dense label matrix
            dscalar: dense scalar matrix

        Returns:
            pscalar: Parcellated scalar matrix
        """

        label_file = nib.load(dlabel_fname)
        dlabel = np.array(label_file.get_fdata()).squeeze()

        labels = np.unique(dlabel)
        N = len(labels)

        pscalar = np.empty((self.data.shape[0], N))
        for ii in range(N):
            pscalar[:, ii] = self.data[:, dlabel == labels[ii]].mean(1)

        if zscore:
            pscalar = stats.zscore(pscalar, axis=1)

        return pscalar

    def set_data(self, data_to_write):
        """
        :param data_to_write: Replaces the values
        :return:
        """
        self._data = data_to_write.reshape(self._data.shape)

    def save(self, fname):
        """

        :param fname: filename to write new file
        :return:
        """
        new_img = nib.cifti2.cifti2.Cifti2Image(self._data, header=self._header)
        nib.save(new_img, fname)

    @property
    def data(self):
        return self._data.squeeze()




class Gifti():
    def __init__(self, fname_lh, fname_rh=None, data = None):
        if data is not None:
            self.input_dir = data.input_dir
            self.output_dir = data.output_dir
        else:
            self.input_dir = ''
            self.output_dir = ''

        self.file_lh = gifti.giftiio.read(self.input_dir+fname_lh)
        if fname_rh is not None:
            self.file_rh = gifti.giftiio.read(self.input_dir+fname_rh)
        else:
            self.file_rh = None


    def data(self, idx):
        if self.file_rh is not None:
            return [self.file_lh.darrays[idx].data, self.file_rh.darrays[idx].data]
        else:
            return self.file_lh.darrays[idx].data


    def set_surf(self):
        if self.file_rh is not None:
            self.vertices_l = self.file_lh.darrays[0].data
            self.triangles_l = self.file_lh.darrays[1].data
            self.vertices_r = self.file_rh.darrays[0].data
            self.triangles_r = self.file_rh.darrays[1].data
        else:
            self.vertices = self.file_lh.darrays[0].data
            self.triangles = self.file_lh.darrays[1].data
