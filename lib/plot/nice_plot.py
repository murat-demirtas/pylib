from .subplot_wrapper import SubPlot
import numpy as np
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .stats import lm
from scipy.stats import pearsonr
import matplotlib as mpl
#import colormaps

class NicePlot(SubPlot):
    def __init__(self, fig, ax):
        super(NicePlot, self).__init__(fig, ax)


    def _multiple_plots(self, x, n, Nf=0.75):
        N = Nf * (x[1] - x[0])  # x-axis extend
        # shift each bar based on number of bars
        if n == 2:
            lag = (-N / n)
        else:
            lag = (-N / n) * (2 - 0.5 * (n % 2))
        d_lag = N / n
        return lag, d_lag


    def _sigstar(self, p):
        if p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''


    def _add_significance(self, x, y, p):
        current_ticks = self.ax.yaxis.get_ticklocs()
        height = 0.02
        for ii in range(p.shape[0]):
            if self._sigstar(p[ii]) is not None:
                self.ax.text(x[ii], y[ii] + height, self._sigstar(p[ii][ii + k]),
                             color=self.almost_black,
                             ha='center', va='center', size=9)

        self.ax.set_yticks(current_ticks)
        self.ax.spines['left'].set_bounds(current_ticks[0], current_ticks[-1])


    def _add_significance2(self, x, y, p):
        current_ticks = self.ax.yaxis.get_ticklocs()

        if x.ndim > 2:
            height = 0.04
            y_max = y[:][0].max()
            for jj in range(p.shape[1]/2 + 1):
                k = jj + 1
                for ii in range(p.shape[1]-k):
                    print(self._sigstar(p[ii][ii+k]))
                    self.ax.plot([x[ii][0],x[ii+k][0]],[y_max+height, y_max+height], color=self.almost_black)
                    self.ax.text((x[ii][0]+x[ii+k][0])/2, y_max+height, '*', color=self.almost_black, ha='center', va='center', size=9)
                    height += 0.04
                height += 0.05
        else:
            height = 0.04
            y_max = y[:].max()
            if not isinstance(p, float):
                for jj in range(p.shape[1] / 2 + 1):
                    k = jj + 1
                    for ii in range(p.shape[1] - k):
                        if self._sigstar(p[ii][ii + k]) is not None:
                            self.ax.plot([x[ii], x[ii + k]], [y_max + height, y_max + height], color=self.almost_black)
                            self.ax.text((x[ii] + x[ii + k]) / 2, y_max + height, self._sigstar(p[ii][ii + k]), color=self.almost_black,
                                         ha='center', va='center', size=9)
                            height += 0.04
                    height += 0.05
            else:
                self.ax.plot([x[0], x[1]], [y_max + height, y_max + height], color=self.almost_black)
                self.ax.text((x[0] + x[1]) / 2, y_max + height, self._sigstar(p),
                             color=self.almost_black,
                             ha='center', va='center', size=9)

        self.ax.set_yticks(current_ticks)
        self.ax.spines['left'].set_bounds(current_ticks[0], current_ticks[-1])


    #===========================================================
    # Simple plotting tools
    #===========================================================
    def plot(self, y, x=None, scatter=False, hide_ax=False,
             colors=None, lines=False, dots=False, line_list=None, marker_list=None,
             palette="Set2", defaults=True, tight=True, *args, **kwargs):

        if defaults:
            kwargs.setdefault('alpha', 0.7)
            kwargs.setdefault('lw', 1.0)
            kwargs.setdefault('mec', self.almost_black)
            kwargs.setdefault('clip_on', False)
            if scatter:
                kwargs.setdefault('ms', 3.0)
            else:
                kwargs.setdefault('ms', 3.0)

        self.despine(full=hide_ax)
        y, x = self.handle_data(y, x, data=None, hue=None)

        # Default line and marker types
        n = len(y) if isinstance(y[0], list) else 1  # Number of plots
        if line_list is None:
            line_list = ['-'] * n  # plain line as default
            if lines: line_list = self.gen_line_list(n)  # default line style list

        if scatter: # The defaults for scatter plot
            line_list = [''] * n # draw no lines
            if marker_list is None:
                marker_list = self.gen_marker_list(n) if dots else ['o'] * n
            #import pdb; pdb.set_trace()
        else:
            if marker_list is None:
                marker_list = [None] * n  # no markers as default
                if dots: marker_list = self.gen_marker_list(n)  # default dot style list


        if isinstance(y[0], list):
            color_list = self.gen_colors(np.arange(n) + 1, palette=palette)[:,:3].tolist() if colors is None else colors
            # for multiple plots
            for ii in range(n):
                self.ax.plot(x[ii], y[ii], ls=line_list[ii],
                             marker=marker_list[ii],
                             color=color_list[ii], *args, **kwargs)
        else:
            # for single plot
            color_list = self.default_color if colors is None else colors
            self.ax.plot(x, y, color=color_list, ls=line_list[0],
                         marker=marker_list[0], *args, **kwargs)

        #if tight: self.tight_ticks()


    def reg_plot(self, y, x, colors_scatter=None, colors_line=None,
                 text_fs=6, text_loc=(0.5, 1.0), text_fw='bold', alpha=0.2,  use_spearman=False, *args, **kwargs):

        #kwargs.setdefault('alpha', 0.2)
        self.plot(y, x, scatter=True, colors=colors_scatter, tight=False, alpha=alpha, *args, **kwargs)
        beta, pval, tval = lm(y, x)
        r, p = pearsonr(x, y)
        self.plot(beta[1] + x*beta[0], x, colors=colors_line, tight=False, alpha=0.7, *args, **kwargs)

        if p < 0.001:
            p_text = '***'
        elif p < 0.01:
            p_text = '**'
        elif p < 0.05:
            p_text = '*'
        else:
            p_text = ''


        if mpl.rcParams['text.usetex']:
            text_result = 'r = ' + '{:3.2f}'.format(r) + p_text
            if use_spearman:
                from scipy.stats import spearmanr
                rs, ps = spearmanr(x, y)
                text_result += '\n' + '$r_{s}$' + ' = {:3.2f}'.format(rs) + p_text
            if text_fw == 'bold':
                text_result = r'\textbf{' + text_result + '}'

        else:
            text_result = 'r = ' + '{:3.2f}'.format(r) + p_text

        self.ax.text(text_loc[0], text_loc[1], text_result, fontsize=text_fs, fontweight=text_fw,
                     horizontalalignment='center', verticalalignment='center',transform=self.ax.transAxes)


        #import pdb; pdb.set_trace()




    def bar(self, y, x=None, x_labels=None,
            orientation='vertical', width=0.75,
            hide_ax=False, colors=None, palette="Set2",
            defaults = True, hatch_list=None, *args, **kwargs):

        self.despine(full=hide_ax)
        if x is None:
            y, x = self.handle_data(y, x, data=None, hue=None)
        if x_labels is None: x_labels = x

        if defaults:
            kwargs.setdefault('alpha', 0.7)

        if isinstance(y[0], list):
            x = np.array(x[0])  # x-axis
            x_ticks = x
            # Case of multiple bars
            n = len(y)  # Number of bars
            color_list = self.gen_colors(np.arange(n) + 1, palette=palette)[:,:3].tolist() if colors is None else colors
            hatch_list = ['']*n if hatch_list is None else hatch_list
            print(hatch_list)
            lag, d_lag = self._multiple_plots(x, n)
            wdt = d_lag * width
            for ii in range(n):
                if orientation=='horizontal':
                    self.ax.barh(x + lag, y[ii], height=wdt, color=color_list[ii], hatch=hatch_list[ii], *args, **kwargs)
                else:
                    self.ax.bar(x + lag, y[ii], width=wdt, color=color_list[ii], hatch=hatch_list[ii], *args, **kwargs)
                lag += d_lag
        else:
            x = np.array(x)
            x_ticks = x + width / 2
            # single bar
            color_list = self.default_color if colors is None else colors
            if orientation == 'horizontal':
                self.ax.barh(x, y, height=width, color=color_list, *args, **kwargs)
            else:
                self.ax.bar(x, y, width=width, color=color_list, *args, **kwargs)

        # Organize ticks and tick-labels
        if orientation == 'vertical':
            self.ax.set_xticks(x_ticks, minor=False)
            if x_labels is not None:
                self.ax.set_xticklabels(x_labels, minor=False)
            else:
                self.ax.set_xticklabels(x, minor=False)
            self.ax.tick_params(axis='x', which='major', pad=0.1)
        else:
            self.ax.set_yticks(x_ticks, minor=False)
            self.ax.set_ylim([x_ticks.min() - 0.5, x_ticks.max() + 0.5])

            if x_labels is not None:
                self.ax.set_yticklabels(x_labels, minor=False)
            else:
                self.ax.set_yticklabels(x, minor=False)
            self.ax.tick_params(axis='y', which='major', pad=0.1)

        #self.tight_ticks()


    def heatmap(self, data, y=None, x=None, n_ticks=None, invert=True,
                clusters=None, legend=False, color_list=None, color_list2=None, labels=None,
                cmap="Blues", cbar=True, cbarloc="right", cbarpad=0.01, cbar_title=None,
                square=True, label_rot=(0,0), xloc='bottom',
                cbar_title_size=10, cbar_title_weight='bold',
                cbartick=3, xsize="2%", labelpad=None,
                **kwargs):


        # Plot matrix as heatmap
        heatmap = self.ax.pcolormesh(data, cmap=cmap, **kwargs)
        if square: self.ax.set_aspect(1)

        if xloc == 'top':
            self.x_axis2top()

        # set axis limits to fit data
        self.ax.set_xlim([0, data.shape[1]])
        self.ax.set_ylim([0, data.shape[0]])
        # by default remove spine
        self.despine(full=True)

        # TODO: add clusters
        # self.ax.add_patch(patches.Rectangle((1,2),10,10,transform=self.ax.transData, fill=False))
        if invert: self.ax.invert_yaxis()
        # Organize ticks and tick-labels
        if y is not None:
            if isinstance(y[0],str):
                if data.shape[0] == len(y):
                    self.ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
                if label_rot[1] != 0:
                    self.ax.set_yticklabels(y, minor=False, rotation=label_rot[1])
                else:
                    self.ax.set_yticklabels(y, minor=False)
                self.ax.tick_params(axis='y', which='major', pad=0.1)
            else:
                tick_index = np.arange(data.shape[0])
                self.ax.set_yticks(tick_index[::n_ticks], minor=False)
                #if labels is None:
                y_labels = ['{:3.1f}'.format(ii) for ii in y[::n_ticks]]
                self.ax.set_yticklabels(y_labels, minor=False)
                #else:
                #    self.ax.set_yticklabels(labels, minor=False)

                #self.ax.set_yticks(y, minor=False)

        if x is not None:
            if isinstance(y[0], str):
                if data.shape[1] == len(x):
                    self.ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
                if label_rot[0] != 0:
                    self.ax.set_xticklabels(x, minor=False, rotation=label_rot[0])
                else:
                    self.ax.set_xticklabels(x, minor=False)
                self.ax.tick_params(axis='x', which='major', pad=0.1)
            else:
                tick_index = np.arange(data.shape[1])
                self.ax.set_xticks(tick_index[::n_ticks], minor=False)
                x_labels = ['{:3.1f}'.format(ii) for ii in x[::n_ticks]]
                self.ax.set_xticklabels(x_labels, minor=False)


        self.ax.tick_params(axis='x', which='major', pad=0.1)
        self.tight_ticks()

        #self.network_legend()
        divider = make_axes_locatable(self.ax)
        heatmap.cmap.set_bad([0.1,0.2,0.3])
        if cbar:
            cmap_mat = heatmap.get_cmap()
            cnorm_mat = heatmap.norm
            self.append_cbar(cmap_mat, cnorm_mat, cbarloc=cbarloc, cbarpad=cbarpad, divider=divider,
                             cbar_title_size=cbar_title_size, cbartick=cbartick, cbar_title_weight=cbar_title_weight,
                             cbar_title=cbar_title, labelpad=labelpad)

        if clusters is not None:
            self.network_axis(clist=color_list, bounds=clusters, labels=labels, xsize=xsize, divider=divider)
            if legend:
                self.network_legend(labels, color_list2, clusters, divider)

    def hmap(self, y, x, data, mask=None, n_ticks=None, invert=True,
                clusters=None, legend=False, color_list=None, color_list2=None, labels=None,
                cmap="Blues", cbar=True, cbarloc="right", cbartick=3,  cbarpad=0.01, cbar_title=None,
                square=False, cbar_title_size=10, cbar_title_weight='bold', shrinked=False, **kwargs):

        if mask is not None:
            data[data == mask] = -np.inf
            data_mask = data#np.ma.masked_invalid(np.atleast_2d(data))
            #import pdb; pdb.set_trace()
            from matplotlib.cm import get_cmap
            cmap_mask = get_cmap(cmap)
            cmap_mask.set_under('w')
            #cmap_mask.set_bad('w', 1.0)
            heatmap = self.ax.pcolormesh(y, x, data_mask, cmap=cmap_mask, **kwargs)
        else:
            # Plot matrix as heatmap
            heatmap = self.ax.pcolormesh(y, x, data, cmap=cmap, **kwargs)
        if square: self.ax.set_aspect(1)
        # by default remove spine
        #self.despine(full=True)

        # self.network_legend()
        divider = make_axes_locatable(self.ax)
        heatmap.cmap.set_bad([0.1, 0.2, 0.3])
        if cbar:
            cmap_mat = heatmap.get_cmap()
            cnorm_mat = heatmap.norm
            self.append_cbar(cmap_mat, cnorm_mat, cbarloc=cbarloc, cbarpad=cbarpad, divider=divider, cbar_title=cbar_title,
                             cbar_title_size=cbar_title_size,  cbartick=cbartick, cbar_title_weight=cbar_title_weight,
                             shrinked=shrinked)


    # ===========================================================
    # Statistical plotting tools
    # ===========================================================
    def boxplot(self, y, x_labels=None,
                strips=True, p_vals=None,
                colors=None, palette='Set2', Nf=3.0, *args, **kwargs):

        self.despine()
        self.ax.spines['bottom'].set_visible(False)
        #self.ax.xaxis.set_ticks_position('none')

        if isinstance(y, list):
            #import pdb; pdb.set_trace()
            if y[0].ndim > 1:
                x = np.arange(y[0].shape[1])+1  # x-axis
                if x_labels is None: x_labels = copy(x)
                # Case of multiple bars
                n = len(y)  # Number of bars
                color_list = self.gen_colors(np.arange(n) + 1, palette=palette)[:,:3].tolist() if colors is None else colors
                lag, d_lag = self._multiple_plots(x, n)
                wdt = d_lag * 0.5
                x_ticks = x - wdt

                boxplt = []
                y_max_whis = np.zeros((n, len(x)))
                x_coord = np.zeros((n, len(x)))
                for ii in range(n):
                    x_positions = x + lag
                    if strips:
                        n_points = y[ii].shape[1]
                        for jj in range(n_points):
                            x_strips = np.random.normal(x_positions[jj], 0.04, y[0].shape[0])
                            self.ax.plot(x_strips, y[ii][:, jj], alpha=0.5, lw=1.0, ls='',
                                         marker='o', ms=1, mec=self.almost_black,
                                         color=color_list[ii], clip_on=False)


                    boxplt += [self.ax.boxplot(y[ii], positions=x + lag, widths=wdt,
                                               showcaps=False, patch_artist=True, sym='', *args, **kwargs)]

                    wind_count = 0
                    for box in boxplt[-1]['boxes']:
                        box.set(facecolor=color_list[ii], edgecolor=self.almost_black, alpha=0.5)
                    for med in boxplt[-1]['medians']:
                        med.set(color=self.almost_black, alpha=0.7)
                    for wind, whis in enumerate(boxplt[-1]['whiskers']):
                        whis.set(color=self.almost_black, alpha=0.7)
                        if wind % 2 == 1:
                            y_max_whis[ii, wind_count] = whis.get_data()[1].max()
                            x_coord[ii, wind_count] = whis.get_data()[0][0]
                            wind_count += 1
                    lag += d_lag
                    if p_vals is not None: self._add_significance2(x_coord, y_max_whis, p_vals)
            else:
                x = (np.arange(len(y)) + 1)/Nf
                if x_labels is None: x_labels = copy(x)
                # Case of multiple bars
                n = len(y)  # Number of bars
                color_list = self.gen_colors(np.arange(n) + 1, palette=palette)[:,
                             :3].tolist() if colors is None else colors

                wdt = 0.1
                x_ticks = x# - wdt

                for ii in range(n):
                    x_positions = x
                    if strips:
                        #n_points = y[ii].shape[0]
                        #for jj in range(y[ii].shape[0]):
                        x_strips = np.random.normal(x_positions[ii], 0.01, y[ii].shape[0])
                        self.ax.plot(x_strips, y[ii], alpha=0.5, lw=1.0, ls='',
                                     marker='o', ms=3, mec=self.almost_black,
                                     color=color_list[ii], clip_on=False)
                boxplt = self.ax.boxplot(y, positions=x, widths=wdt,
                                         showcaps=False, patch_artist=True, sym='', *args, **kwargs)

                y_max_whis = np.zeros(len(x))
                x_coord = np.zeros(len(x))
                wind_count = 0
                count = 0
                #import pdb; pdb.set_trace()
                for box in boxplt['boxes']:
                    box.set(facecolor=color_list[count], edgecolor=self.almost_black, alpha=0.5)
                    count += 1
                for med in boxplt['medians']:
                    med.set(color=self.almost_black, alpha=0.7)
                for wind, whis in enumerate(boxplt['whiskers']):
                    whis.set(color=self.almost_black, alpha=0.7)
                    if wind % 2 == 1:
                        y_max_whis[wind_count] = whis.get_data()[1].max()
                        x_coord[wind_count] = whis.get_data()[0][0]
                        wind_count += 1
                if p_vals is not None: self._add_significance2(x_coord, y_max_whis, p_vals)

        else:
            x_ticks = np.arange(y.shape[1]) + 1  # x-axis
            if x_labels is None: x_labels = x_ticks
            color_list = self.default_color
            if strips:
                for ii in range(y.shape[1]):
                    x = np.random.normal(ii + 1, 0.04, y.shape[0])
                    self.ax.plot(x, y[:, ii], alpha=0.3, lw=1.0, ls='', marker='o', ms=1, mec=self.almost_black, color=color_list, clip_on=False)

            boxplt = self.ax.boxplot(y, widths=0.25,
                                     showcaps=False, patch_artist=True, sym='', *args, **kwargs)
            y_max_whis = np.zeros(len(x))
            x_coord = np.zeros(len(x))
            wind_count = 0
            for box in boxplt['boxes']:
                box.set(facecolor=color_list, edgecolor=self.almost_black, alpha=0.5)
            for med in boxplt['medians']:
                med.set(color=self.almost_black, alpha=0.7)
            for wind, whis in enumerate(boxplt['whiskers']):
                whis.set(color=self.almost_black, alpha=0.7)

                if wind % 2 == 1:
                    y_max_whis[wind_count] = whis.get_data()[1].max()
                    x_coord[wind_count] = whis.get_data()[0][0]
                    wind_count += 1
            if p_vals is not None: self._add_significance2(x_coord, y_max_whis, p_vals)

        self.ax.set_xticks(x_ticks, minor=False)
        self.ax.set_xticklabels(x_labels, minor=False)
        self.ax.tick_params(axis='x', which='major', pad=0.1)


    def distplot(self, y, bins, colors=None, palette='Set2', kde=False, ho=False, *args, **kwargs):
        self.despine()
        if kde:
            n = len(y)
            color_list = self.gen_colors(np.arange(n) + 1, palette=palette)[:,:3].tolist() if colors is None else colors
            from scipy.stats import gaussian_kde
            density = gaussian_kde(y[0])
            pdf = density(bins)
            pdf = pdf / pdf.sum()
            if ho:
                self.ax.fill_between(pdf, bins, color=color_list, facecolor=color_list, linewidth=0.0, alpha=0.5)
                self.ax.plot(pdf, bins, color=color_list, *args, **kwargs)

            else:
                self.ax.fill(bins, pdf, color=color_list, fc=color_list, *args, **kwargs)
        else:
            n = len(y)
            color_list = self.gen_colors(np.arange(n) + 1, palette=palette)[:, :3].tolist() if colors is None else colors
            self.ax.hist(y, bins, color=color_list, *args, **kwargs)


    def pointplot(self, y=None, x=None, colors=None,
                  sem=True, fill=True, palette="Set2",
                  p_vals=None, *args, **kwargs):

        self.despine()
        if isinstance(y, list):
            n = len(y)
            if colors is None:
                color_list_1 = self.gen_colors(np.arange(n) + 1, palette=palette)[:, :3].tolist()
            else:
                color_list_1 = colors

            for ii in range(n):
                if x is None: x = np.arange(1, y[0].shape[0] + 1)
                n_obs = y[ii].shape[1]
                y_mean = y[ii].mean(1)
                error = y[ii].std(1)
                if sem: error /= np.sqrt(n_obs)
                color_list = np.array(color_list_1[ii])
                if fill:
                    self.ax.plot(x, y_mean, color=color_list / 2.0, clip_on=False, *args, **kwargs)
                    self.ax.fill_between(x, y_mean - error, y_mean + error, color=color_list, alpha=0.5)
                else:
                    marker_list = ['.']
                    ms = 5
                    self.ax.errorbar(x, y_mean, yerr=error, color=color_list, ecolor=color_list, marker=marker_list[0],
                                     ms=ms,
                                     mec=color_list / 2.0, mfc=color_list, capsize=None)
        else:
            if x is None: x = np.arange(1, y.shape[0] + 1)
            n_obs = y.shape[1]
            y_mean = y.mean(1)
            error = y.std(1)
            if sem: error /= np.sqrt(n_obs)

            if colors is None:
                color_list = self.default_color
            else:
                color_list = np.array(colors)

            if fill:
                self.ax.plot(x, y_mean, color=color_list/2.0, clip_on=False, *args, **kwargs)
                self.ax.fill_between(x, y_mean-error, y_mean+error,  color=color_list, alpha=0.5)
            else:
                marker_list = ['.']
                ms = 5
                self.ax.errorbar(x, y_mean, yerr = error, color=color_list, ecolor=color_list, marker=marker_list[0], ms=ms,
                                 mec=color_list/2.0, mfc=color_list, capsize=None)

            if p_vals is not None: self._add_significance(x, y_mean+error, p_vals)

    def set_brain(self, surface='inflated', parc='cole', subcortex=False, mayavi=False,
                  cifti_extension=None, cifti_template=None, scene_template=None, dpi=300):
        if mayavi:
            from brain import Brain
            self.brain = Brain(self.ax, surface=surface, parc=parc, subcortex=subcortex)
        else:
            from brain_wb import Brain
            self.brain = Brain(self.ax, cifti_extension, cifti_template, scene_template=scene_template, dpi=dpi)


    def brain_cbar(self, cbar_loc=None, cbar_title='', inset1=None, orientation=None, append=False):
        if cbar_loc == 'left':
            inset1 = {'w': '2%',
                      'h': '20%',
                      'loc': 4}
            orientation = 'vertical'
        elif cbar_loc == 'right':
            inset1 = {'w': '2%',
                      'h': '20%',
                      'loc': 5}
            orientation = 'vertical'
        elif cbar_loc == 'top':
            inset1 = {'w': '20%',
                      'h': '2%',
                      'loc': 9}
            orientation = 'horizontal'
        elif cbar_loc == 'bottom':
            inset1 = {'w': '20%',
                      'h': '2%',
                      'loc': 8}
            orientation = 'horizontal'
        else:
            if inset1 is None:
                inset1 = {'w': '25%',
                          'h': '3%',
                          'loc': 10}
            if orientation is None:
                orientation = 'horizontal'

        if append:
            norm = mpl.colors.Normalize(vmin=self.brain.vrange[0], vmax=self.brain.vrange[1])
            self.append_cbar(self.brain.cmap, norm=norm, cbarloc=cbar_loc, cbarpad=0.05, divider=None, cbar_title=cbar_title,
                        cbar_title_size=8, cbar_title_weight='normal', cbartick=2)
        else:
            self.add_cbar(cmap=self.brain.cmap, range=self.brain.vrange, orientation=orientation, inset=inset1, title=cbar_title)

    def set_circle(self):
        self.convert2polar()
        from circle_plot import Circle
        self.circle = Circle(self.fig, self.ax)


    def circle_cbar(self, cbar_loc='left', cbar_title='', inset1=None, orientation=None):
        if cbar_loc == 'left':
            inset1 = {'w': '20%',
                      'h': '2%',
                      'loc': 3}
            orientation = 'horizontal'
        elif cbar_loc == 'right':
            inset1 = {'w': '20%',
                      'h': '2%',
                      'loc': 4}
            orientation = 'vertical'
        elif cbar_loc == 'bottom':
            inset1 = {'w': '20%',
                      'h': '2%',
                      'loc': 8}
            orientation = 'horizontal'
        else:
            inset1 = {'w': '25%',
                      'h': '3%',
                      'loc': 10}
            orientation = 'horizontal'

        self.add_cbar(cmap=self.circle.cmap, range=self.circle.vrange, orientation=orientation, inset=inset1,
                      title=cbar_title)

    def image(self, fname):
        import matplotlib.image as mpimg
        img = mpimg.imread(fname)
        self.ax.imshow(img)
        self.ax.axis('off')
