from lxml import etree
import matplotlib.image as mpimg
from os import system
from utils.cifti import Cifti
import os

class Brain():
    def __init__(self, ax, cifti_extension, cifti_template, scene_template=None, dpi=300):
        self.ax = ax
        self.surf_templates = {'inflated':'Conte69_inflated_32k',
                  'veryinflated':'Conte69_veryinflated_32k',
                  'anatomical':'Conte69_midthickness_32k'}

        self.scenes = {
                  'montage':2,
                  'left':3,
                  'right':4
                  }

        self.resolution = {
                  'montage':[dpi*8,dpi*6],
                  'left':[dpi*8, dpi*3],
                  'rigth':[dpi*8, dpi*3]
                  }

        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.parent_path = os.path.abspath(os.path.join(self.module_path, os.pardir))
        self.template_dir = self.parent_path+'/data/templates/templates_32k/'
        self.temp_dir = self.module_path + '/temp/'

        if scene_template is None:
            doc = etree.parse(self.template_dir+'template_32k.scene')
        else:
            doc = etree.parse(scene_template)
        self.elem = doc.getroot()

        self.parcel_templates = {'dscalar':
                                 {'dscalar': self.template_dir+'cifti/template_32k.dscalar.nii'},

                                 'pscalar':
                                 {'glasser': self.template_dir+'cifti/glasser_with_subcortex.pscalar.nii',
                                 'aparc': self.template_dir+'cifti/aparc.pscalar.nii',
                                 'yeo': self.template_dir+'cifti/yeo_17Networks.pscalar.nii'},

                                 'dlabel':
                                 {'glasser': self.template_dir+'cifti/glasser_with_subcortex.dlabel.nii',
                                  'aparc': self.template_dir+'cifti/aparc.dlabel.nii',
                                  'yeo': self.template_dir+'cifti/yeo_17Networks.dlabel.nii'}}

        self.cifti_templates = {'dscalar': 'template_32k.dscalar.nii',
                                'pscalar': 'template_32k.pscalar.nii',
                                'dlabel': 'template_32k.dlabel.nii'}

        self.cifti_template = self.cifti_templates[cifti_extension]
        self.ext = cifti_extension

        if self.parcel_templates[cifti_extension].has_key(cifti_template):
            self.cifti = Cifti(self.parcel_templates[cifti_extension][cifti_template])
        else:
            self.cifti = Cifti(cifti_template)

        self.palette_changed = False


    def set_borders(self, line_width, template):
        if template == 'glasser':
            border_idx = [14, 15, 16, 17]
        else:
            border_idx = [16, 17, 14, 15]
        dgroup = self.elem[self.scene_id][2][0][2][0][0][4][0][0].text

        if dgroup == 'DISPLAY_GROUP_TAB':
            self.elem[self.scene_id][2][0][2][0][0][4][1][0].text = 'true'
            self.elem[self.scene_id][2][0][2][0][0][4][3][0].text = str(line_width)
            self.elem[self.scene_id][2][0][2][0][0][border_idx[0]][1][0].text = 'false'
            self.elem[self.scene_id][2][0][2][0][0][border_idx[1]][1][0].text = 'false'
            #self.elem[self.scene_id][2][0][2][0][0][border_idx[2]][5][2][0].text = 'true'
            #self.elem[self.scene_id][2][0][2][0][0][border_idx[3]][5][2][0].text = 'true'
        else:
            self.elem[self.scene_id][2][0][2][0][0][4][5][1].text = 'true'
            self.elem[self.scene_id][2][0][2][0][0][4][7][1].text = str(line_width)
            self.elem[self.scene_id][2][0][2][0][0][border_idx[0]][3][1].text = 'false'
            self.elem[self.scene_id][2][0][2][0][0][border_idx[1]][3][1].text = 'false'

        # TODO: Select border
        #for ii in range(5,5+len(border)):
        #    if border[ii-5] == 0:
        #        self.elem[self.scene_id][2][0][2][0][0][16][5][ii][1][0].text = 'false'

    def set_surface_template(self, surf, show_cbar):
        n_models = len(self.elem[self.scene_id][2][0][2][0][0][12])
        for m in range(n_models):
            m_type = self.elem[self.scene_id][2][0][2][0][0][12][m][0][0].text
            if m_type == 'MODEL_TYPE_WHOLE_BRAIN':
                if self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][0][0][1][1][0][0][1].text == 'true':
                    n_surf = len(self.elem[self.scene_id][2][0][2][0][0][12][m][0][2])
                    for ii in range(n_surf):
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][2][ii].text = self.surface
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][5].text = self.cifti_template
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][6].text = self.cifti_template
                        if not show_cbar:
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][4][13].text = 'false'
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][4][14].text = 'false'

                    n_surf = len(self.elem[self.scene_id][2][0][2][0][0][12][m][0][8])
                    for ii in range(n_surf):
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][8][ii][0][2].text = self.surface
                        if self.elem[self.scene_id][2][0][2][0][0][12][m][0][8][ii][0][1].text == 'CORTEX_LEFT':
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][8][ii][0][3].text = surf + '_L.surf.gii'
                        else:
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][8][ii][0][3].text = surf + '_R.surf.gii'
            elif m_type == 'MODEL_TYPE_SURFACE_MONTAGE':
                if self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][0][0][1][1][0][0][1].text == 'true':
                    n_surf = len(self.elem[self.scene_id][2][0][2][0][0][12][m][0][5])
                    for ii in range(n_surf):
                        if not show_cbar:
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][4][13].text = 'false'
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][4][14].text = 'false'
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][5][ii][0][1][1][0][0][4][13].text = 'false'
                            self.elem[self.scene_id][2][0][2][0][0][12][m][0][5][ii][0][1][1][0][0][4][14].text = 'false'
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][5].text = self.cifti_template
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][1][ii][0][1][1][0][0][6].text = self.cifti_template

                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][5][ii][0][1][1][0][0][5].text = self.cifti_template
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][5][ii][0][1][1][0][0][6].text = self.cifti_template
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][5][ii][0][2][0].text = surf + '_L.surf.gii'
                        self.elem[self.scene_id][2][0][2][0][0][12][m][0][5][ii][0][4][0].text = surf + '_R.surf.gii'
        #import pdb; pdb.set_trace()
        #self.elem[self.scene_id][2][0][2][0][0][12][6][0][2][0][0][1][1][0][0][5].get('Name')

    def set_cmap(self, palette):
        self.cifti.set_cmap(palette=palette)
        self.palette_changed = True

    def set_cmap_volume(self):
        for ii in range(len(self.elem[self.scene_id][2][0][2][0][0][0])):
            if len(self.elem[self.scene_id][2][0][2][0][0][0][ii][0]) > 3:
                if len(self.elem[self.scene_id][2][0][2][0][0][0][ii][0][2]) > 0:
                    if self.elem[self.scene_id][2][0][2][0][0][0][ii][0][2][0][0].get('Name') == 'savedPaletteColorMapping':
                        self.elem[self.scene_id][2][0][2][0][0][0][ii][0][2][0][0][3].text = self.cifti.extensions[0][1][0][1][0][1].text


    def set_cbar_legend(self, legend):
        annotations = etree.fromstring(self.elem[self.scene_id][2][0][2][0][0][2][1].text)

        if (self.cifti_template=='template_32k.dlabel.nii')|(legend is None):
            annotations[1].remove(annotations[1][0])
        else:
            annotations[1][0][1].text = legend
        self.elem[self.scene_id][2][0][2][0][0][2][1].text = etree.tostring(annotations)

    def add_annotation(self, text, coord, properties):
        # TODO: groupType='STEREOTAXIC', 'SURFACE', 'WINDOW', 'rotationAngle'...etc.
        text_props = {'color': 'textCaretColor',
                      'bold': 'fontBold',
                      'italic':'fontItalic',
                      'ha':'horizontalAlignment',
                      'va':'verticalAlignment',
                      'orientation':'orientation',
                      'font_size': 'fontPercentViewportSize'}
        annotations = etree.fromstring(self.elem[self.scene_id][2][0][2][0][0][2][1].text)
        buffer = etree.fromstring(self.elem[self.scene_id][2][0][2][0][0][2][1].text)[1][0]
        annotations[1].append(buffer)
        annotations[1][-1][1].text = text
        annotations[1][-1][0].set('x', str(coord[0]))
        annotations[1][-1][0].set('y', str(coord[1]))
        annotations[1][-1][0].set('z', str(coord[2]))
        for key in properties.keys():
            if text_props.has_key(key):
                annotations[1][-1][1].set(text_props[key], properties[key])
        self.elem[self.scene_id][2][0][2][0][0][2][1].text = etree.tostring(annotations)

    def make_scene(self):
        file = open(self.template_dir+'temp.scene','w')
        file.write(etree.tostring(self.elem,pretty_print=True))

    def plot(self, data=None, scene='montage', surface='inflated', legend=None,
             cmap='RdBu_r', vrange=None, cthreshold=None,
             border = False, show_cbar=True, border_lw = 3, border_temp = 'glasser',
             annotations = None, hemi_label=False, title=None):
        self.scene_id = self.scenes[scene]

        if annotations is not None:
            # TODO: annotation list, other annotation types
            self.add_annotation(annotations['text'], annotations['coord'], annotations)

        if border:
            self.set_borders(line_width=border_lw, template=border_temp)

        if data is not None:
            if self.ext == 'dlabel':
                self.cifti.dlabel_cmap(data, cmap_name=cmap, vrange=vrange, threshold=cthreshold)
                self.cmap = cmap
                self.vrange = self.cifti.vrange
            else:
                self.cifti.set_data(data)
                self.cmap = cmap
                self.vrange = vrange

        if self.palette_changed:
            self.set_cmap_volume()

        self.cifti.save(self.template_dir+self.cifti_template)
        self.surface = surface.upper()
        self.set_surface_template(self.surf_templates[surface], show_cbar)

        self.set_cbar_legend(legend)
        self.make_scene()

        cmd = 'wb_command -show-scene ' + self.template_dir + 'temp.scene ' + scene +  ' ' + self.temp_dir + 'output.png ' + str(int(self.resolution[scene][0])) + ' ' + str(int(self.resolution[scene][1]))
        system(cmd)
        img = mpimg.imread(self.temp_dir+'output.png')
        self.ax.imshow(img)
        self.ax.axis('off')

        if hemi_label:
            self.ax.text(0.0, 1.0, 'L', transform=self.ax.transAxes,
                         fontsize=10, fontweight='bold', va='top', ha='left')

            self.ax.text(1.0, 1.0, 'R', transform=self.ax.transAxes,
                         fontsize=10, fontweight='bold', va='top', ha='right')

        if title is not None:
            self.ax.text(0.5, 1.0, title, transform=self.ax.transAxes,
                         va='bottom', ha='center')


        system('rm ' + self.template_dir + 'temp.scene')
        system('rm ' + self.temp_dir + 'output.png')


