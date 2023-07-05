#! usr/bin/python
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.font_manager as font_manager

# Internal libraries
from .nice_plot import NicePlot

class Figure(object):
    def __init__(self, size=(4,2.5), dpi=100, latex=False, params=None):
        #font_dirs = ['/Users/md2242/Library/Fonts', ]
        #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        #font_list = font_manager.createFontList(font_files)
        #font_manager.fontManager.ttflist.extend(font_list)

        self.panel = plt.figure(figsize=size, dpi=dpi, facecolor='white')
        self._figure = []
        self.figure = []
        self._divider = []

        self.brain_flag = True

        default_params = {
            'tick_fonts': 8.0,
            'label_fonts': 10.0,
            'title_fonts': 10.0
            }

        from matplotlib import rc
        #rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        #rc('font', **{'family': 'Helvetica'})
        #======================================
        # Define new defaults
        # ======================================
        mpl.rcParams['axes.edgecolor'] = '#262626'

        #mpl.rcParams['font.family'] = 'sans-serif'
        #mpl.rcParams['font.sans-serif'] = 'Helvetica'
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 10.0
        mpl.rcParams['axes.titlesize'] = 10.0
        mpl.rcParams['xtick.labelsize'] = 10.0
        mpl.rcParams['ytick.labelsize'] = 10.0
        mpl.rcParams['ytick.major.size'] = 2
        mpl.rcParams['xtick.major.size'] = 2
        mpl.rcParams['ytick.major.pad'] = 1
        mpl.rcParams['xtick.major.pad'] = 1
        mpl.rcParams['ps.useafm'] = True
        mpl.rcParams['pdf.use14corefonts'] = True
        mpl.rcParams['xtick.direction'] = 'out'
        mpl.rcParams['ytick.direction'] = 'out'
        mpl.rcParams['legend.fontsize'] = 8
        mpl.rcParams['legend.frameon'] = False

        if params is not None:
            for key in params.keys():
                mpl.rcParams[key] = params[key]

        if latex:
            #import pdb; pdb.set_trace()
            #plt.rc('text', usetex=True)
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.unicode'] = True
            #mpl.rcParams['font.sans-serif'] = ['Helvetica']
            mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{helvet}",
                                                   r'\usepackage{sansmath}', r'\sansmath']

        #from colormaps import Cm
        #self.perceptual_colormaps = Cm()


    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            x = pos[0]
            y = pos[1]
        else:
            x = pos
            y = 0
        return self.figure[x][y]

    def subplot(self, grid=11, margins=(0.06,0.06), scale=0.95, polar=None):
        shape = np.ones(2)
        shape[0] = float(str(grid)[0]); shape[1] = float(str(grid)[1])
        l_cols = (0.95 - margins[1])/shape[0]
        l_rows = (0.95 - margins[0])/shape[1]
        idx = 0
        for cols in range(int(shape[0])):
            for rows in range(int(shape[1])):
                self.add_plot([margins[0]+cols*l_cols/scale, margins[1]+rows*l_rows/scale, l_cols*scale**(1.0/l_cols), l_rows*scale**(1.0/l_rows)], polar=False)


    def add_plot(self, loc, *args, **kwargs):
        self._figure += [self.panel.add_axes([loc[0],loc[1],loc[2],loc[3]], *args, **kwargs)]
        self.figure += [[]]
        self._divider += [[]]
        self.figure[-1] += [NicePlot(self.panel, self._figure[-1])]


    def add_subaxis(self, axis, loc, size, pad, sharex=False, sharey=False):
        #if isinstance(self._divider[axis], empty):
        if len(self._divider[axis]) == 0:
            self._divider[axis] = [make_axes_locatable(self.figure[axis][0].ax)]
        if sharex:
            new_axis = self._divider[axis][0].append_axes(loc, size=size, pad=pad, sharex=self.figure[axis][0].ax)
        elif sharey:
            new_axis = self._divider[axis][0].append_axes(loc, size=size, pad=pad, sharey=self.figure[axis][0].ax)
        else:
            new_axis = self._divider[axis][0].append_axes(loc, size=size, pad=pad)
        self.figure[axis] += [NicePlot(self.panel, new_axis)]

    def add_twin(self, axis_id):
        new_axis = self.figure[axis_id][0].ax.twinx()
        #new_axis.yaxis.tick_right()
        self.figure[axis_id] += [NicePlot(self.panel, new_axis)]


    def sharex(self, labeltext):
        self.panel.text(0.025, 0.5, labeltext, va='center',ha='center',rotation='vertical', fontsize=12, fontweight='bold')


    def sharey(self, labeltext):
        self.panel.text(0.5, 0.025, labeltext, va='center', ha='center', fontsize=12, fontweight='bold')


    def sharet(self, labeltext):
        self.panel.text(0.5, 0.975, labeltext, va='center', ha='center', fontsize=12, fontweight='bold')

    def show(self):
        plt.show()



