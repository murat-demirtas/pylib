import nibabel as nib
import numpy as np
import nibabel.gifti as gifti
from scipy import stats

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
