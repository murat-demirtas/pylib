import matplotlib.image as mpimg
from os import system
import nibabel as nib
import os

def plot_brain(ax, data=None, pscalar='default',
               scene='pscalar_lr',
               hemi_label=True, title=None):

    module_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.abspath(os.path.join(module_path, os.pardir))
    template_dir = parent_path + '/data/templates/templates_32k/'
    if pscalar == 'default':
        input_template = template_dir + 'template_default_32k.pscalar.nii'
    else:
        input_template = pscalar

    if data is not None:
        pdata = nib.load(input_template)
        data_dummy = data.reshape(pdata.get_fdata().shape)
        gbc_image = nib.cifti2.cifti2.Cifti2Image(data_dummy, header=pdata.header)
        nib.save(gbc_image, template_dir + 'template_32k.pscalar.nii')

    dpi = 300
    resolution = [dpi * 6, dpi * 4]

    cmd = 'wb_command -show-scene ' + template_dir + 'template_32k_simple.scene ' + scene + ' ' + template_dir + 'output.png ' + str(
        int(resolution[0])) + ' ' + str(int(resolution[1]))
    system(cmd)
    img = mpimg.imread(template_dir + 'output.png')
    ax.imshow(img)
    ax.axis('off')

    if hemi_label:
        ax.text(0.05, 0.9, 'L', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')

        ax.text(0.95, 0.9, 'R', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='right')

    if title is not None:
        ax.text(0.5, 0.95, title, transform=ax.transAxes,
                va='bottom', ha='center')

    system('rm ' + template_dir + 'output.png')
