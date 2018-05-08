from lxml import etree
import xml.etree.cElementTree as ET
import nibabel as nib
import numpy as np
from matplotlib import cm
import matplotlib.colors as clrs
import nibabel.gifti as gifti

class Cifti():
    def __init__(self, fname, data = None):
        if data is not None:
            self.input_dir = data.input_dir
            self.output_dir = data.output_dir
        else:
            self.input_dir = ''
            self.output_dir = ''

        of = nib.load(self.input_dir + fname)
        self.data = of.get_data()
        self.affine = of.get_affine()
        self.header = of.get_header()

        self.palette_names = ['PSYCH','PSYCH-NO-NONE','ROY-BIG','ROY-BIG-BL','Orange-Yellow','Gray_Interp_Positive','Gray_Interp','clear_brain','videen_style',
                        'fidl','raich4_clrmid','raich6_clrmid','HSB8_clrmid','RBGYR20','RBGYR20P','POS_NEG','red-yellow','blue-lightblue','FSL','power_surf','fsl_red',
                        'fsl_green','fsl_blue','fsl_yellow','JET256']

        self.palette_modes = ['MODE_AUTO_SCALE','MODE_AUTO_SCALE_ABSOLUTE_PERCENTAGE','MODE_AUTO_SCALE_PERCENTAGE','MODE_USER_SCALE']

        self.palette_map = {'ScaleMode': 0, 'AutoScalePercentageValues': 1, 'AutoScaleAbsolutePercentageValues': 2, 'UserScaleValues': 3, 'PaletteName': 4,
                        'InterpolatePalette': 5, 'DisplayPositiveData': 6, 'DisplayZeroData': 7, 'DisplayNegativeData': 8, 'ThresholdTest': 9, 'ThresholdType': 10,
                        'ThresholdFailureInGreen': 11, 'ThresholdNormalValues': 12, 'ThresholdMappedValues': 13, 'ThresholdMappedAvgAreaValues': 14, 'ThresholdDataName': 15,
                        'ThresholdRangeMode': 16,'ThresholdLowHighLinked': 17, 'NumericFormatMode': 18, 'PrecisionDigits': 19, 'NumericSubivisions': 20,
                        'ColorBarValuesMode': 21, 'ShowTickMarksSelected': 22}

        self.extensions = ET.fromstring(self.header.extensions[0].get_content())
        self.ischanged = False


    def write_extensions(self):
        self.header.extensions[0]._content = ET.tostring(self.extensions)


    def set_map(self, map_name):
        self.extensions[0][1][0][0].text = map_name
        self.ischanged = True


    def set_cmap(self, palette):
        cmap = etree.fromstring(self.extensions[0][1][0][1][0][1].text)
        for key in palette.keys():
            cmap[self.palette_map[key]].text = palette[key]
        self.extensions[0][1][0][1][0][1].text = etree.tostring(cmap)
        self.ischanged = True


    def dlabel_cmap(self, data, cmap_name, vrange=None, threshold=None):
        self.vrange = [data.min(), data.max()] if vrange is None else vrange

        cNorm = clrs.Normalize(vmin=self.vrange[0], vmax=self.vrange[1])

        clr_map = cm.ScalarMappable(cmap=cmap_name, norm=cNorm)
        colors = clr_map.to_rgba(data)
        if threshold is not None:
            crop_idx = data < threshold
            ncrop = np.sum(crop_idx)
            if ncrop > 0:
                nullmap = cm.Greys(0.2 * np.ones(ncrop))
                colors[crop_idx,:] = nullmap
        else:
            nullmap = cm.Greys(0.2 * np.ones(np.sum(data==0.0)))
            colors[data==0.0,:] = nullmap

        for ii in range(1,len(self.extensions[0][1][0][1])):
            self.extensions[0][1][0][1][ii].set('Red', str(colors[ii-1,0]))
            self.extensions[0][1][0][1][ii].set('Green', str(colors[ii-1,1]))
            self.extensions[0][1][0][1][ii].set('Blue', str(colors[ii-1,2]))
            self.extensions[0][1][0][1][ii].set('Alpha', str(colors[ii-1,3]))
        self.ischanged = True


    def dlabel_make_cmap(self, colors):
        for ii in range(1,len(self.extensions[0][1][0][1])):
            self.extensions[0][1][0][1][ii].set('Red', str(colors[ii-1,0]))
            self.extensions[0][1][0][1][ii].set('Green', str(colors[ii-1,1]))
            self.extensions[0][1][0][1][ii].set('Blue', str(colors[ii-1,2]))
            self.extensions[0][1][0][1][ii].set('Alpha', str(colors[ii-1,3]))
        self.ischanged = True


    def get_labels(self, left=True, right=True):
        stx = False
        dlabels = np.squeeze(self.data)
        label_info = {}
        if len(self.extensions[0][2]) < 3:
            l_model = 0
            r_model = 1
        else:
            stx = True
            l_model = 1
            r_model = 2

        if left:
            n_vertices = int(self.extensions[0][2][l_model].get('SurfaceNumberOfVertices'))

            left_offset = int(self.extensions[0][2][l_model].get('IndexOffset'))
            left_idx_count = int(self.extensions[0][2][l_model].get('IndexCount'))

            l_idx_str = (self.extensions[0][2][l_model][0].text).split()
            l_indices = [int(ii) for ii in l_idx_str]

            left_scalars = dlabels[left_offset:(left_offset + left_idx_count)]
            left_labels = np.unique(left_scalars)
            left_surf = np.zeros(n_vertices)
            left_surf[l_indices] = left_scalars

            left_scalars_array = np.zeros(n_vertices)
            for ii, ss in enumerate(left_labels):
                left_scalars_array[left_surf == ss] = ii + 1

            label_info['L'] = {}

            label_info['L']['labels'] = left_labels
            label_info['L']['scalars'] = left_surf
            label_info['L']['scalars_array'] = left_scalars_array

        if right:
            n_vertices = int(self.extensions[0][2][r_model].get('SurfaceNumberOfVertices'))

            right_offset = int(self.extensions[0][2][r_model].get('IndexOffset'))
            right_idx_count = int(self.extensions[0][2][r_model].get('IndexCount'))

            r_idx_str = (self.extensions[0][2][r_model][0].text).split()
            r_indices = [int(ii) for ii in r_idx_str]

            right_scalars = dlabels[right_offset:(right_offset + right_idx_count)]
            right_labels = np.unique(right_scalars)
            right_surf = np.zeros(n_vertices)
            right_surf[r_indices] = right_scalars

            right_scalars_array = np.zeros(n_vertices)
            for ii, ss in enumerate(right_labels):
                right_scalars_array[right_surf == ss] = ii + 1

            label_info['R'] = {}

            label_info['R']['labels'] = right_labels
            label_info['R']['scalars'] = right_surf
            label_info['R']['scalars_array'] = right_scalars_array

        #if stx:
        #    n_regions = len(self.extensions[0][2])-2
        #    self.extensions[0][2][3][0].text.split('\n')[0].split() # coords
        #    import pdb; pdb.set_trace()

        return label_info

    def stx_offset(self):
        return int(self.extensions[0][2][3].get('IndexOffset'))



    def set_pdata(self, data, dlabel):
        parcels = nib.load(dlabel).get_data().squeeze()
        data_to_write = np.empty(parcels.shape)
        #import pdb; pdb.set_trace()
        for ii, pp in enumerate(np.unique(parcels)):
            data_to_write[parcels==pp] = data[ii]
        self.set_data(data_to_write)


    def set_data(self, data_to_write):
        self.data = data_to_write.reshape(self.data.shape)


    def save(self, fname):
        if self.ischanged: self.write_extensions()
        new_img = nib.Nifti2Image(self.data,affine=self.affine,header=self.header)
        nib.save(new_img, self.output_dir+fname)


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
