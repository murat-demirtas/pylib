import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
#from mpl_toolkits.axes_grid.inset_locator import inset_axes

class SubPlot(object):
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.index = len(self.fig.axes) - 1

        self.corr = 1.0

        self.almost_black = '#262626'
        self.default_color = np.array([67.0, 162.0, 202.0])/255.0


    #=====================================================
    # Handle data: use list as default.
    #              define x-axis value if necessary
    #=====================================================
    def handle_data(self, y, x=None, data=None, hue=None):
        if data is None:
            if x is None:
                if not isinstance(y, list): y = y.tolist()
                if isinstance(y[0],list):
                    n = len(y)
                    x = []
                    for ii in range(n):
                        x += [[]]
                        x[-1] = range(1,len(y[ii])+1)
                else:
                    x = range(1,len(y)+1)
            else:
                if not isinstance(y, list): y = y.tolist()
                if not isinstance(x, list): x = x.tolist()
                if isinstance(y[0],list):
                    if not isinstance(x[0], list):
                        x = [x]*len(y)

        return y, x


    #============================================
    # Generate marker and line styles, and colors
    #============================================
    def gen_line_list(self, N):
        ls_list = ['-', '--', ':', '-.']
        return np.tile(ls_list, (1, N/4 + 1))[0][:N]


    def gen_marker_list(self, N):
        ms_list = ['o', '^', 's', 'p', 'h', 'x', '+', '8']
        return np.tile(ms_list, (1, N / 8 + 1))[0][:N]


    def gen_colors(self, data, palette, vrange=None):
        if vrange is None: vrange = [data.min(), data.max()]
        cmap = plt.get_cmap(palette)
        cNorm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        return scalarMap.to_rgba(data)


    #=====================================================
    # Title, axis-labels and caption_id (A, B, C...etc.)
    # ====================================================
    def title(self, title_string, *args, **kwargs):
        self.ax.set_title(title_string, *args, **kwargs)


    def xlabel(self, label_string, labelpad = 1, *args, **kwargs):
        self.ax.set_xlabel(label_string, labelpad=labelpad, *args, **kwargs)


    def ylabel(self, label_string, labelpad = 2, *args, **kwargs):
        self.ax.set_ylabel(label_string, labelpad=labelpad, *args, **kwargs)


    def caption_id(self, labeltext, padx=0.075, pady=1.1, fsize=14, *args, **kwargs):
        self.ax.text(padx, pady, labeltext,
                     transform=self.ax.transAxes,
                     va='center',ha='center', fontsize=fsize, fontweight='bold', *args, **kwargs)


    def caption_id_fig(self, labeltext, padx=0.075, pady=1.1, fsize=14, *args, **kwargs):
        self.ax.text(padx, pady, labeltext,
                     transform=self.fig.transFigure,
                     va='center',ha='center', fontsize=fsize, fontweight='bold', *args, **kwargs)



    #=====================================================
    # Axis properties:
    #=====================================================
    def convert2polar(self):
        from matplotlib import projections

        x0 = self.ax.get_position().x0
        y0 = self.ax.get_position().y0# + 0.075
        x1 = self.ax.get_position().x1# - 0.05
        y1 = self.ax.get_position().y1# - 0.05

        rect = (x0, y0, x1-x0, y1-y0)

        projection_class, kwa, key = projections.process_projection_requirements(self.fig, polar=True)
        key = ((rect,), key[1])
        self.ax = projection_class(self.fig, rect, **kwa)
        self.fig._axstack.remove(self.fig.axes[self.index])
        self.fig._axstack.add(key, self.ax)

    def remove_x_axis(self):
        self.ax.get_xaxis().set_visible(False)


    def remove_y_axis(self):
        self.ax.get_yaxis().set_visible(False)


    def invert_axis(self, axis='y'):
        if axis == 'y':
            self.ax.invert_yaxis()
        else:
            self.ax.invert_xaxis()


    def x_axis2top(self):
        self.ax.xaxis.tick_top()


    def tick_out(self, ax='both'):
        self.ax.tick_params(axis=ax, which='major', direction='out')


    def no_ticks(self, yaxis=True, xaxis=True, nolabels=False):
        if xaxis:
            self.ax.xaxis.set_ticks_position('none')
            if nolabels:
                self.ax.set_xticklabels([''])
        if yaxis:
            self.ax.yaxis.set_ticks_position('none')
            if nolabels:
                self.ax.set_yticklabels([''])

    def set_spine(self, loc='right', color='k'):
        if loc == 'right':
            self.ax.spines["right"].set_visible(True)
            self.ax.spines["left"].set_visible(False)
            self.ax.yaxis.tick_right()
            self.ax.tick_params('y', colors=color)
            self.ax.spines["right"].set_color(color)
        else:
            self.ax.tick_params('y', colors=color)
            self.ax.spines["left"].set_color(color)


    def despine(self, full=False):
        # Remove top and right axes lines ("spines")
        spines_to_remove = ['top', 'right']
        for spine in spines_to_remove:
            self.ax.spines[spine].set_visible(False)

        if full:
            self.no_ticks()
            spines_to_remove = ['bottom', 'left']
            for spine in spines_to_remove:
                self.ax.spines[spine].set_visible(False)
        else:
            spines_to_keep = ['bottom', 'left']
            for spine in spines_to_keep:
                self.ax.spines[spine].set_linewidth(0.5)
                self.ax.spines[spine].set_color(self.almost_black)
            self.ax.xaxis.tick_bottom()
            self.ax.yaxis.tick_left()


    def shift_yaxis(self, y_plus):
        current_ticks = self.ax.yaxis.get_ticklocs()
        self.ax.set_ylim((current_ticks[0], current_ticks[-1]+y_plus))
        self.ax.set_yticks(current_ticks)

    def detach(self, axis='both', p=0.02, spine='left'):
        if axis == 'y':
            current_ticks = self.ax.yaxis.get_ticklocs()
            range = current_ticks[-1] - current_ticks[0]
            shift = range*p
            self.ax.set_ylim(current_ticks[0] - shift, current_ticks[-1] + shift)
            self.ax.set_yticks(current_ticks)
            self.ax.spines[spine].set_bounds(current_ticks[0], current_ticks[-1])

        if axis == 'x':
            current_ticks = self.ax.xaxis.get_ticklocs()
            range = current_ticks[-1] - current_ticks[0]
            shift = range * p
            self.ax.set_xlim(current_ticks[0] - shift, current_ticks[-1] + shift)
            self.ax.set_xticks(current_ticks)
            self.ax.spines['bottom'].set_bounds(current_ticks[0], current_ticks[-1])

        if axis == 'both':
            current_yticks = self.ax.yaxis.get_ticklocs()
            yrange = current_yticks[-1] - current_yticks[0]
            yshift = yrange * p
            self.ax.set_ylim(current_yticks[0] - yshift, current_yticks[-1] + yshift)
            self.ax.set_yticks(current_yticks)
            self.ax.spines[spine].set_bounds(current_yticks[0], current_yticks[-1])

            current_xticks = self.ax.xaxis.get_ticklocs()
            xrange = current_xticks[-1] - current_xticks[0]
            xshift = xrange*p
            self.ax.set_xlim(current_xticks[0] - xshift, current_xticks[-1] + xshift)
            self.ax.set_xticks(current_xticks)
            self.ax.spines['bottom'].set_bounds(current_xticks[0], current_xticks[-1])

            #import pdb; pdb.set_trace()

    def rotate_tick_labels(self, axis, rotation=90):
        if axis == 'x':
            lbl = [self.ax.get_xticklabels()[ii].get_text().encode('utf-8')
                   for ii in range(len(self.ax.get_xticklabels()))]

            self.ax.set_xticklabels(lbl, rotation=rotation)
        if axis == 'y':
            lbl = [self.ax.get_yticklabels()[ii].get_text().encode('utf-8')
                   for ii in range(len(self.ax.get_yticklabels()))]

            self.ax.set_yticklabels(lbl, rotation=rotation)


    def gen_ticks(self, tmin, tmax, n):
        trange = tmax - tmin

        if trange >= 1.0:
            step = trange / (n - 1)
            if step >= 1.0:
                precision = 0
                tmin = np.around(tmin, precision)
                tmax = np.around(tmax, precision)
                trange = tmax - tmin
                if trange % (n - 1) == 0.0:
                    ticks = np.around(np.linspace(tmin, tmax, n), precision)
                else:
                    alt_step = np.arange(2, n + 1)
                    test = trange % alt_step
                    if any(test == 0):
                        new_n = alt_step[np.where(test == 0)[0][-1]]
                    else:
                        if np.abs(2 - trange) < np.abs(n - trange):
                            new_n = 1
                        else:
                            new_n = trange
                    ticks = np.around(np.linspace(tmin, tmax, new_n + 1), precision)
            else:
                precision = int(np.ceil(-np.log10(step)))
                tmin = np.around(tmin, precision)
                tmax = np.around(tmax, precision)
                trange = tmax - tmin
                actual_step = int((10 ** precision) * trange)
                if actual_step % (n - 1) == 0.0:
                    ticks = np.around(np.linspace(tmin, tmax, n), precision)
                else:
                    alt_step = np.arange(2, n + 1)
                    test = actual_step % alt_step
                    if any(test == 0):
                        new_n = alt_step[np.where(test == 0)[0][-1]]
                    else:
                        if np.abs(2 - actual_step) < np.abs(n - actual_step):
                            new_n = 1
                        else:
                            new_n = actual_step
                    ticks = np.around(np.linspace(tmin, tmax, new_n + 1), precision)
        else:
            step = trange / (n - 1)
            precision = int(np.ceil(-np.log10(step)))

            tmin = np.around(tmin, precision)
            tmax = np.around(tmax, precision)
            trange = tmax - tmin

            actual_step = int((10 ** precision) * trange)
            if actual_step % (n - 1) == 0.0:
                ticks = np.around(np.linspace(tmin, tmax, n), precision)
            else:
                alt_step = np.arange(2, n + 1)
                test = actual_step % alt_step
                if any(test == 0):
                    new_n = alt_step[np.where(test == 0)[0][-1]]
                else:
                    if np.abs(2 - actual_step) < np.abs(n - actual_step):
                        new_n = 1
                    else:
                        new_n = actual_step
                ticks = np.around(np.linspace(tmin, tmax, new_n + 1), precision)
        return ticks


    def ytick_count(self, n):
        tmin = self.ax.get_ylim()[0]
        tmax = self.ax.get_ylim()[1]
        if n < 3:
            ticks = np.array([tmin, tmax])
        else:
            ticks = self.gen_ticks(tmin, tmax, n)
        #print ticks
        self.ax.yaxis.set_ticks(ticks)

    def xtick_count(self, n):
        tmin = self.ax.get_xlim()[0]
        tmax = self.ax.get_xlim()[1]
        if n < 3:
            ticks = np.array([tmin, tmax])
        else:
            ticks = self.gen_ticks(tmin, tmax, n)
        self.ax.xaxis.set_ticks(ticks)

    def set_ytick_format(self, f = '%3.1f'):
        self.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f))

    def set_xtick_format(self, f = '%3.1f'):
        self.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f))

    def set_yticks(self, ticks, ticks_string = None, f = '{:3.1f}'):
        self.ax.set_yticks(ticks, minor=False)
        if ticks_string is None:
            y_labels = [f.format(ii) for ii in ticks]
        else:
            y_labels = ticks_string
        self.ax.set_yticklabels(y_labels, minor=False)

    def set_xticks(self, ticks, ticks_string = None, f = '{:3.1f}'):
        self.ax.set_xticks(ticks, minor=False)
        if ticks_string is None:
            x_labels = [f.format(ii) for ii in ticks]
        else:
            x_labels = ticks_string
        self.ax.set_xticklabels(x_labels, minor=False)

    def tight_ticks(self):
        xticks = self.ax.get_xticklabels()
        yticks = self.ax.get_yticklabels()

        bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= 80
        height *= 80

        optimal_xlabel_size = width / len(xticks)
        if optimal_xlabel_size < 8.0:
            [xticks[ii].set_size(self.corr*optimal_xlabel_size) for ii in range(len(xticks))]

        optimal_ylabel_size = width / len(yticks)
        if optimal_ylabel_size < 8.0:
            [yticks[ii].set_size(self.corr*optimal_ylabel_size) for ii in range(len(yticks))]

    # =====================================================
    # Legend wrapper:
    # =====================================================
    def hline(self, xloc, yloc=None, *args, **kwargs):
        if yloc is None:
            ylims = self.ax.yaxis.axes.get_ylim()
        else:
            ylims = yloc
        self.ax.plot([xloc, xloc], ylims, *args, **kwargs)
        self.ax.set_ylim(ylims)

    def vline(self, yloc, xloc=None, *args, **kwargs):
        if xloc is None:
            xlims = self.ax.xaxis.axes.get_xlim()
        else:
            xlims = xloc
        self.ax.plot(xlims, [yloc, yloc], *args, **kwargs)
        self.ax.set_xlim(xlims)


    #=====================================================
    # Legend wrapper:
    #=====================================================
    def add_legend(self, legends, anchor=None, tight=True, *args, **kwargs):
        if anchor is None:
            kwargs.setdefault('bbox_to_anchor', (1.0, 1.0))
        else:
            kwargs.setdefault('bbox_to_anchor', anchor)
        if tight:
            kwargs.setdefault('markerscale', 0.75)
            kwargs.setdefault('handlelength', 0.75)
            kwargs.setdefault('borderpad', 0.1)
            kwargs.setdefault('borderaxespad', 0.05)
            kwargs.setdefault('scatterpoints', 2)
            kwargs.setdefault('labelspacing', 0.15)
            kwargs.setdefault('handletextpad', 0.25)

        self.ax.legend(legends, *args, **kwargs)

    def network_legend(self, labels, clist, clusters, divider=None, ncols=2):
        if divider is None:
            size = 1.0
            ax = self.ax
        else:
            size = 0.2
            ax = divider.append_axes('top', size=str(100*size)+'%', pad=0.)

        import matplotlib.patches as mpatches
        rect = []
        nrows = np.ceil(float(len(labels))/float(ncols))
        h = .5/(nrows)
        w = h*1.2#*size

        kk = 0
        for jj, lab in enumerate(labels):

            x = 0.0 + (1./ncols)*np.mod(jj,ncols)
            y = h/2. + (h*2.)*np.floor(jj/ncols)

            #rect += [mpatches.Rectangle((x, y), width=w, height=h, facecolor=clist[clusters[jj],:3],
            #                         edgecolor='black', linewidth=1, label=labels[jj],transform=ax.transAxes)]

            rect += [mpatches.Rectangle((x, y), width=w, height=h, facecolor=clist[jj,:3],
                                    edgecolor='black', linewidth=1, label=lab,transform=ax.transAxes)]

            ax.add_patch(rect[-1])
            #ax.text(x + w*1.2, y + h/2., lab, transform=ax.transAxes, fontsize=8, va='center', ha='left')
            ax.text(x + w * 1.1, y + h / 2., lab, transform=ax.transAxes, fontsize=8, va='center', ha='left', fontweight='bold')

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)


    #=====================================================
    # Standalone Colorbar
    #=====================================================
    def append_cbar(self, cmap, norm=None, cbarloc='left', cbarpad=0.05, divider=None, cbar_title=None,
                    cbar_title_size=6, cbar_title_weight='normal', cbartick=3,
                    shrinked=False, labelpad=None, anchor=None, *args, **kwargs):

        if cbarloc == 'right':
            orientation = 'vertical'
            anchor = (0.0, 1.0)
            shrink = 0.6
        elif cbarloc == 'left':
            orientation = 'vertical'
            anchor = (0.0, 1.0)
            shrink = 0.6
        elif cbarloc == 'bottom':
            orientation = 'horizontal'
            anchor = (0.5, 0.0)
            shrink = 1.0
        else:
            orientation = 'horizontal'
            anchor = (0.5, 0.0)
            shrink = 1.0



        if shrinked:
            from matplotlib.colorbar import make_axes
            cax, kw = make_axes(self.ax, location=cbarloc, fraction=0.05,
                            pad=cbarpad, shrink=shrink, anchor=anchor)
        else:
            if divider is None:
                divider = make_axes_locatable(self.ax)
            cax = divider.append_axes(cbarloc, size="5%", pad=cbarpad)


        #import pdb;
        #pdb.set_trace()
        #import pdb; pdb.set_trace()ColorbarBase
        self.cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation=orientation,
                                            ticks=np.linspace(norm.vmin, norm.vmax, cbartick).tolist(), *args, **kwargs)

        #self.cb.ax.set_position([0.05, 0.44, 0.33, 0.5])
        #import pdb; pdb.set_trace()
        if cbar_title is not None:
            if labelpad is None:
                self.cb.set_label(cbar_title, weight=cbar_title_weight, size=cbar_title_size)  # , labelpad=30)
            else:
                self.cb.set_label(cbar_title, weight=cbar_title_weight, size=cbar_title_size,
                                  labelpad=labelpad)  # , labelpad=30)

            #self.cb.ax.set_title(cbar_title, weight=cbar_title_weight, size=cbar_title_size)  # , labelpad=30)
        self.cb.ax.xaxis.set_ticks_position('none')
        self.cb.ax.yaxis.set_ticks_position('none')
        self.cb.ax.tick_params(labelsize=6)
        self.cb.outline.set_visible(False)


    def add_cbar(self, cmap, range, orientation='horizontal', title=None, inset=None):
        norm = mpl.colors.Normalize(vmin=range[0], vmax=range[1])
        if inset is not None:
            cax = inset_axes(self.ax, width=inset['w'], height=inset['h'], loc=inset['loc'])
        else:
            cax = self.ax

        self.cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation=orientation,
                                           ticks=[range[0], range[1]])

        self.cb.ax.xaxis.set_ticks_position('none')
        self.cb.ax.yaxis.set_ticks_position('none')
        self.cb.ax.tick_params(labelsize=6)
        self.cb.ax.xaxis.set_ticks_position('none')
        self.cb.outline.set_visible(False)

        if title is not None:
            if orientation == 'horizontal':
                self.cb.ax.text(0.5, 1.2, title,
                               fontsize=6, fontweight='bold', va='bottom', ha='center')
            else:
                self.cb.ax.text(-0.2, 0.5, title, rotation=90,
                                fontsize=6, fontweight='bold', va='center', ha='right')


    def network_axis(self, clist, bounds, labels=None, divider=None, vbar='left', hbar='top',
                     xsize="2%", *args, **kwargs):
        #self.remove_x_axis(); self.remove_y_axis()
        #self.ax.xaxis.set_ticks_position('none')
        #self.ax.yaxis.set_ticks_position('none')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        from matplotlib import colors
        if divider is None:
            divider = make_axes_locatable(self.ax)

        cmap = colors.ListedColormap(clist)
        norm = mpl.colors.Normalize(vmin=min(bounds), vmax=max(bounds))


        cax_left = divider.append_axes(vbar, size=xsize, pad=0)
        self.cb_left = mpl.colorbar.ColorbarBase(cax_left, cmap=cmap, norm=norm, orientation='vertical', *args, **kwargs)
        self.cb_left.ax.xaxis.set_ticks_position('none')
        self.cb_left.ax.yaxis.set_ticks_position('none')
        self.cb_left.ax.get_yaxis().set_ticks([])
        self.cb_left.ax.invert_yaxis()
        self.cb_left.outline.set_visible(False)

        cax_bottom = divider.append_axes(hbar, size=xsize, pad=0)
        self.cb_bottom = mpl.colorbar.ColorbarBase(cax_bottom, cmap=cmap, norm=norm, orientation='horizontal', *args, **kwargs)
        self.cb_bottom.ax.xaxis.set_ticks_position('none')
        self.cb_bottom.ax.yaxis.set_ticks_position('none')
        self.cb_bottom.ax.get_xaxis().set_ticks([])
        self.cb_bottom.outline.set_visible(False)


