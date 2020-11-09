from mayavi import mlab
import numpy as np
from copy import copy
from matplotlib import cm
import matplotlib.colors as clrs
import os
from ..analysis.cifti import Gifti, Cifti

class Brain():
    def __init__(self, surface='veryinflated', parc='glasser'):
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.parent_path = os.path.abspath(os.path.join(self.module_path, os.pardir))
        self.template_path = self.parent_path + '/data/templates/templates_32k/'

        surface_gifti = 'surface/Conte69_' + surface
        surf_file = Gifti(self.template_path + surface_gifti + '_32k_L.surf.gii',
                          self.template_path + surface_gifti + '_32k_R.surf.gii')
        self.vertices = {'L': surf_file.data(0)[0],
                         'R': surf_file.data(0)[1]}
        self.triangles = {'L': surf_file.data(1)[0],
                          'R': surf_file.data(1)[1]}
        self.mni2ijk()

        if parc is None:
            self.scalars_array = {'L': np.arange(1, self.vertices['L'].shape[0] + 1),
                                  'R': np.arange(1, self.vertices['L'].shape[0] + 1)}
            self.groups = {'L': np.arange(self.vertices['L'].shape[0]),
                           'R': np.arange(self.vertices['L'].shape[0], 2 * (self.vertices['L'].shape[0]))}
        else:
            if parc == 'glasser':
                parc_file = self.template_path+'cifti/Glasser_NetworkPartition_v9_2.dlabel.nii'
            elif parc == 'aparc':
                parc_file = self.template_path + 'cifti/aparc.dlabel.nii'
            elif parc == 'yeo':
                parc_file = self.template_path + 'cifti/yeo_17Networks.dlabel.nii'
            else:
                parc_file = self.template_path + 'cifti/' + parc + '.dlabel.nii'

            parcel = Cifti(parc_file)
            labels = parcel.data
            label_info = parcel.get_labels()

            self.scalars = {}
            self.scalars['L'] = label_info['L']['scalars']
            self.scalars['R'] = label_info['R']['scalars']

            self.scalars_array = {}
            self.scalars_array['L'] = label_info['L']['scalars_array']
            self.scalars_array['R'] = label_info['R']['scalars_array']

            scalars_cortex = np.unique(np.hstack((label_info['L']['labels'], label_info['R']['labels'])))
            hemis = np.zeros(len(scalars_cortex))  # left: 0, right: 1

            for ii in range(len(label_info['R']['labels'])):
                idx = np.where(scalars_cortex == label_info['R']['labels'][ii])
                hemis[idx] = 1.0

            self.groups = {'L': np.where(hemis == 0.0)[0],
                           'R': np.where(hemis == 1.0)[0]}

        zoom = [170.0, 180.0, 200.0, 200.0, 225.0, 225.0]
        self.views = {'L':
                          {'medial': [0.0, -90.0, zoom[0]],
                           'sagittal': [0.0, 90.0, zoom[1]],
                           'ventral': [90.0, 90.0, zoom[2]],
                           'dorsal': [-90.0, 90.0, zoom[3]],
                           'top': [0.0, 0.0, zoom[4]],
                           'bottom': [0.0, 180.0, zoom[5]]},
                      'R':
                          {'medial': [0.0, 90.0, zoom[0]],
                           'sagittal': [0.0, -90.0, zoom[1]],
                           'ventral': [90.0, 90.0, zoom[2]],
                           'dorsal': [-90.0, 90.0, zoom[3]],
                           'top': [0.0, 0.0, zoom[4]],
                           'bottom': [0.0, 180.0, zoom[5]]}}
        self.hemi_sign = {'L': 1.0, 'R': -1}

    def mni2ijk(self, fs_average=True):
        '''
        Transform coordinates to fit the surface and volume data
        '''
        # TODO: extract transformation information from surface file
        if fs_average:
            N = 2.0 # adjust downsampling
            a = np.eye(3)
            affine_vect = np.array([1.0, -135.0, 126.0, 72.0])/N
            # Left hemisphere
            self.vertices['L'] = self.vertices['L'] * affine_vect[0] + affine_vect[1] * a[0] + affine_vect[2] * a[1] + affine_vect[3] * a[2]
            #self.vertices['L'][:, 0] *= -1.0
            # Right hemisphere
            affine_vect = np.array([1.0, -45.0, 126.0, 72.0])/N
            self.vertices['R'] = self.vertices['R'] * affine_vect[0] + affine_vect[1] * a[0] + affine_vect[2] * a[1] + affine_vect[3] * a[2]
            #self.vertices['R'][:, 0] *= -1.0

    def set_data(self, data):
        '''
        Given the data, create colormaps
        '''
        self.colormap = np.floor(255 * cm.Greys(0.2 * np.ones(len(data))))
        self.data_range = {'L': [-1, self.scalars_array['L'].max()],
                           'R': [-1, self.scalars_array['R'].max()]}

        #import pdb; pdb.set_trace()
        nullcolor = copy(self.colormap[:2,:])
        if self.vrange is None:
            self.datamin = data.min()
            self.datamax = data.max()
            self.vrange = [data.min(), data.max()]

        cNorm = clrs.Normalize(vmin=self.vrange[0], vmax=self.vrange[1])

        clr_map = cm.ScalarMappable(cmap=self.cmap, norm=cNorm)
        colors = np.floor(255 * clr_map.to_rgba(data))
        if self.threshold is not None:
            crop_idx = data < self.threshold
            ncrop = np.sum(crop_idx)
            if ncrop > 0:
                nullmap = np.floor(255 * cm.Greys(0.2 * np.ones(ncrop)))
                colors[crop_idx,:] = nullmap
        else:
            nullmap = np.floor(255 * cm.Greys(0.2 * np.ones(np.sum(data==0.0))))
            colors[data==0.0,:] = nullmap

        self.colormap = colors
        self.ctx_clr = {'L': np.vstack((nullcolor, self.colormap[self.groups['L'],:])),
                        'R': np.vstack((nullcolor, self.colormap[self.groups['R'],:]))}


    def cortex(self, vertex, triangle, scalar, colors, datarange, fig, hemi):
        '''
        Plot cortical surface
        '''
        surface = mlab.triangular_mesh(vertex[:, 0], vertex[:, 1], vertex[:, 2], triangle, colormap='Set2',
                                       scalars=scalar, figure=fig)

        # Add borders
        if self.borders is not None:
            test = self.borders[self.groups[hemi]].sum()
            if test > 0:
                for ii in np.where(self.borders[self.groups[hemi]])[0]:
                    scalar_dummy = np.zeros(scalar.shape)
                    scalar_dummy[scalar == ii + 1] = 1.0
                    s = mlab.pipeline.triangular_mesh_source(vertex[:, 0], vertex[:, 1], vertex[:, 2], triangle, scalars=scalar_dummy)
                    cont = mlab.pipeline.contour_surface(s, contours=2)
                    cont.actor.mapper.interpolate_scalars_before_mapping = True
                    cont.actor.property.line_width = 2.0

        surface.module_manager.scalar_lut_manager.lut.table = colors
        surface.module_manager.scalar_lut_manager.use_default_range = False
        surface.module_manager.scalar_lut_manager.data_range = datarange

    def surface(self, hemi, view=None, dpi=100, size=None, use_tvtk = False):
        '''
        Plot surface
        '''

        # set canvas size, if not specified
        if size is None:
            if view == 'ventral' or view == 'dorsal':
                size = (4.0*dpi, 5.0*dpi)
            elif view == 'top' or view == 'bottom':
                size = (3.0*dpi, 4.0*dpi)
            elif view == 'medial' or view == 'sagittal':
                size = (4.0 * dpi, 3.0 * dpi)
            else:
                size = (4.0 * dpi, 3.0 * dpi)

        mlab.options.offscreen = True
        fig = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=size)
        fig.scene.disable_render = True
        
        if use_tvtk:
            from tvtk.api import tvtk
            rw = tvtk.RenderWindow(size=fig.scene._renwin.size, off_screen_rendering=1)
            rw.add_renderer(fig.scene._renderer)

        if hemi == 'both':
            vertex_L = self.transform(self.vertices['L'], [1.0, -2.5, -5.0, 7.5])
            vertex_R = self.transform(self.vertices['R'], [1.0, 2.5, -5.0, 7.5])

            self.cortex(vertex_L, self.triangles['L'], self.scalars_array['L'], self.ctx_clr['L'], self.data_range['L'], fig, 'L')
            self.cortex(vertex_R, self.triangles['R'], self.scalars_array['R'], self.ctx_clr['R'], self.data_range['L'], fig, 'R')
            
            mlab.view(self.views['L'][view][0], self.views['L'][view][1], self.views['L'][view][2])
        else:
            vertex = self.vertices[hemi]
            self.cortex(vertex, self.triangles[hemi], self.scalars_array[hemi], self.ctx_clr[hemi],
                              self.data_range[hemi],
                              fig, hemi)

            if isinstance(view, str):
                mlab.view(self.views[hemi][view][0], self.views[hemi][view][1], self.views[hemi][view][2])
            else:
                mlab.view(view[0], view[1], view[2])

        if use_tvtk:
            w2if = tvtk.WindowToImageFilter()
            w2if.input = rw
            w2if.update()
            
        fig.scene.disable_render = False
        self.image += [mlab.screenshot(fig)]

    def montage(self, style, dpi=100):
        '''
        Create montages
        '''
        self.image = []
        if style == 'transverse':
            self.surface(hemi='both', view='top', dpi=dpi)
            self.surface(hemi='both', view='bottom', dpi=dpi)
            img = np.hstack((self.image[0], self.image[1]))
        elif style == 'default':
            self.surface(hemi='L', view='medial', dpi=dpi)
            self.surface(hemi='R', view='medial', dpi=dpi)
            self.surface(hemi='L', view='sagittal', dpi=dpi)
            self.surface(hemi='R', view='sagittal', dpi=dpi)
            img_l = np.vstack((self.image[0], self.image[2]))
            img_r = np.vstack((self.image[1], self.image[3]))
            img = np.hstack((img_l, img_r))
        elif style == 'left':
            self.surface(hemi='L', view='medial', dpi=dpi)
            self.surface(hemi='L', view='sagittal', dpi=dpi)
            img = np.vstack((self.image[1], self.image[0]))
        elif style == 'right':
            self.surface(hemi='R', view='medial', dpi=dpi)
            self.surface(hemi='R', view='sagittal', dpi=dpi)
            img = np.vstack((self.image[1], self.image[0]))
        else:
            self.surface(hemi=style[0], view=style[1], dpi=dpi)
            img = self.image[0]
        return img


    def plot(self, ax, data,
             montage='default', cmap='RdBu_r',
             threshold = None, vrange = None,
             hemi_label = True, title = None,
             borders = None,
             labels = None, label_orientation = None,
             latex = False,
             dpi=100):
        '''
        Plotting script
        '''

        self.ax = ax
        self.cmap = cmap
        self.label_ind = labels
        self.borders = borders
        self.threshold = threshold
        self.vrange = vrange
        self.label_orientation = label_orientation
        self.set_data(data)

        img = self.montage(montage, dpi)
        self.ax.imshow(img)
        self.ax.axis('off')

        bbox = self.ax.get_window_extent()
        factor = bbox.width / 175.0

        if hemi_label:
            if latex:
                self.ax.text(0.0, 1.0, r'\textbf{L}', transform=self.ax.transAxes,
                             fontsize = 5, fontweight='bold', va='top', ha='left')

                self.ax.text(1.0, 1.0, r'\textbf{R}', transform=self.ax.transAxes,
                             fontsize = 5, fontweight='bold', va='top', ha='right')
            else:
                self.ax.text(0.0, 1.0, 'L', transform=self.ax.transAxes,
                 fontsize=6 * factor, fontweight='bold', va='top', ha='left')

                self.ax.text(1.0, 1.0, 'R', transform=self.ax.transAxes,
                             fontsize = 6 * factor, fontweight='bold', va='top', ha='right')
            

        if title is not None:
            self.ax.text(0.5, 1.0, title, transform=self.ax.transAxes,
                         fontsize= 8 * factor, va='bottom', ha='center')
