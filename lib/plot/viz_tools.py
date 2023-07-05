from lib.plot.scifig import Figure
from matplotlib.pyplot import cm
from scipy.stats import spearmanr, pearsonr
import numpy as np

class Fig():
    def __init__(self, size, dpi=300):
        params = {'legend.fontsize': 6.0, 'font.size': 6.0, 'axes.labelsize': 6.0,
                  'axes.titlesize': 6.0, 'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0}

        self.fig = Figure(size=size, dpi=dpi, latex=True, params=params)

        colors_set2 = cm.Set2(np.linspace(0, 1, 8))[:, :3]
        self.clr_emp = colors_set2[7, :]

        colors_set1 = cm.Set1(np.linspace(0, 1, 9))[:, :3]
        self.clr_hom = colors_set1[1, :]
        self.clr_het = colors_set1[0, :]
        self.clr_het[0] = 0.9
        self.clr_het[1] = 0.28
        self.clr_het[2] = 0.1
        #self.clr_emp = colors_try[7, :]

        #self.network_labels = ['VIS', 'AUD', 'SOM', 'VAN', 'FPN', 'DAN', 'CON', 'DMN']
        self.network_labels = ['AUD', 'VIS', 'SOM', 'DAN', 'FPN', 'VAN', 'DMN', 'CON']

    def design(self, design):
        for ii in range(len(design)):
            self.fig.add_plot(design[ii])

    def add_text(self, idx, loc, text, clr, fs=6, transform=True, rotation=0):
        if transform:
            self.fig[idx].ax.text(loc[0], loc[1], text, color=clr, fontsize=fs, fontweight='normal',
                                 verticalalignment='center',
                                 horizontalalignment='center',
                                  transform=self.fig[idx].ax.transAxes, rotation=rotation)
        else:
            self.fig[idx].ax.text(loc[0], loc[1], text, color=clr, fontsize=fs, fontweight='normal',
                                  verticalalignment='center',
                                  horizontalalignment='center', rotation=rotation)


    def plot_brain(self, idx, mask, cmap, clabel=None, vrange=None,
                   border=None, labelpad=-5, hemil=True, title='', prec='{:.1f}'):
        if vrange is None:
            vrange = [mask.min(), mask.max()]

        '''
        brain_mask = np.hstack((np.zeros(19), mask))
        self.fig[idx].set_brain(cifti_extension='dlabel', cifti_template='glasser', dpi=100)
        self.fig[idx].brain.plot(brain_mask, cmap=cmap, hemi_label=False, border=border, title=title, vrange=vrange)
        self.fig[idx].brain_cbar(cbar_loc='bottom', append=True, cbar_title=None)
        if clabel is not None:
            self.fig[idx].cb.set_label(clabel, size=6, labelpad=labelpad)
            self.fig[idx].cb.ax.set_xticklabels(['{:.1f}'.format(vrange[0]), '{:.1f}'.format(vrange[1])], fontsize=6)
        '''
        brain_mask = mask#np.hstack((np.zeros(19), mask))
        #brain_plot = Brain(ax, surface='inflated', parc='cole')
        self.fig[idx].set_brain(surface='veryinflated', parc='cole', mayavi=True, dpi=300)
        self.fig[idx].brain.plot(brain_mask, cmap=cmap,
                                 hemi_label=hemil, borders=border, title=title, vrange=vrange)
        self.fig[idx].brain_cbar(cbar_loc='bottom', append=True, cbar_title=None)
        if clabel is not None:
            self.fig[idx].cb.set_label(clabel, size=6, labelpad=labelpad)
            self.fig[idx].cb.ax.set_xticklabels([prec.format(vrange[0]), prec.format(vrange[1])], fontsize=6)


    def plot_brain_left(self, idx, mask, cmap,
                        clabel=None, vrange=None, border=None, labelpad=-5,
                        hemil=False, title='', cbar=True):
        if vrange is None:
            vrange = [mask.min(), mask.max()]

        '''
        brain_mask = np.hstack((np.zeros(19), mask))
        self.fig[idx].set_brain(cifti_extension='dlabel', cifti_template='glasser',  dpi=100)
        self.fig[idx].brain.plot(brain_mask,  cmap=cmap, scene='left', hemi_label=False, border=border, title=title, vrange=vrange)
        self.fig[idx].brain_cbar(cbar_loc='bottom', append=True, cbar_title=None)
        if clabel is not None:
            self.fig[idx].cb.set_label(clabel, size=6, labelpad=labelpad)
            self.fig[idx].cb.ax.set_xticklabels(['{:.1f}'.format(vrange[0]), '{:.1f}'.format(vrange[1])], fontsize=6)
        '''
        brain_mask = mask
        #brain_plot = Brain(ax, surface='inflated', parc='cole')
        self.fig[idx].set_brain(surface='veryinflated', parc='cole', mayavi=True, dpi=300)
        self.fig[idx].brain.plot(brain_mask, cmap=cmap, style='left',
                                 hemi_label=hemil, borders=border, title=title, vrange=vrange)
        if cbar:
            self.fig[idx].brain_cbar(cbar_loc='bottom', append=True, cbar_title=None)
        if clabel is not None:
            self.fig[idx].cb.set_label(clabel, size=6, labelpad=labelpad)
            self.fig[idx].cb.ax.set_xticklabels(['{:.1f}'.format(vrange[0]), '{:.1f}'.format(vrange[1])], fontsize=6)



    def plot_parameters(self, idx, x, cat, ylims, title, grad, islabel=False):
        if cat == 'homogeneous':
            clr = self.clr_hom
        elif cat == 'heterogeneous':
            clr = self.clr_het
        else:
            clr = self.clr_emp


        #import pdb; pdb.set_trace()
        if cat == 'homogeneous':
            self.fig[idx].pointplot(x, colors=clr, sem=False)
        else:
            self.fig[idx].pointplot(x, grad, colors=clr, sem=False)

        self.fig[idx].ax.set_ylim([ylims[0], ylims[-1]])
        self.fig[idx].ax.set_yticks(ylims)


        self.fig[idx].xtick_count(2)
        #self.fig[idx].ax.set_xticklabels(['1', '180'])

        self.fig[idx].detach()
        #self.fig[idx].title(title)
        self.fig[idx].ylabel(title)
        if isinstance(islabel, str):
            self.fig[idx].xlabel(islabel)
        else:
            self.fig[idx].ax.set_xticklabels([''])
            #self.fig[idx].no_ticks(yaxis=False, nolabels=True)

        if cat == 'heterogeneous':
            self.fig[idx].ax.set_yticklabels([''])
            #self.fig[idx].no_ticks(xaxis=False, nolabels=True)
            #self.fig[idx].ax.spines["left"].set_visible(False)

    def plot_scatter_dist(self, idx, x, y, cat, kde=True):
        if cat == 'homogeneous':
            clr = self.clr_hom
        elif cat == 'heterogeneous':
            clr = self.clr_het
        else:
            clr = self.clr_emp

        self.fig[idx].reg_plot(y, x,
                        colors_scatter=clr, colors_line=[0.0, 0.0, 0.0],
                        text_fs=6, text_loc=(0.5, 1.0), mec=clr, mew=0.0, ms=1.2, lw=0.8)


        self.fig.add_subaxis(idx, 'right', '22%', 0.1, sharey=True)
        self.fig.add_subaxis(idx, 'top', '22%', 0.1, sharex=True)

        if kde:
            self.fig[(idx, 1)].distplot([y], np.linspace(-0.2, 1.0, 50), colors=self.clr_emp, kde=True,
                                        ho=True, lw=0.5, alpha=0.8)
            self.fig[(idx, 2)].distplot([x], np.linspace(-0.05, 1.0, 50), colors=clr, kde=True, lw=0.5, alpha=0.8)
        else:
            self.fig[(idx, 1)].distplot([y], np.linspace(0.0, 1.0, 30), colors=self.clr_emp,
                                 orientation='horizontal', lw=0.5, alpha=0.8)
            self.fig[(idx, 2)].distplot([x], np.linspace(0.0, 1.0, 30), colors=clr, lw=0.5, alpha=0.8)

        self.fig[idx].ax.set_ylim([-0.2, 1.0])
        self.fig[idx].ax.set_xlim([0.0, 1.0])

        self.fig[idx].ax.set_yticks([-0.2, 0.2, 0.6, 1.0])
        self.fig[idx].ax.set_xticks([0.0, 0.5, 1.0])

        #self.fig[idx].xtick_count(6)
        #self.fig[idx].ytick_count(7)

        self.fig[idx].xlabel('Model FC', size=6)
        #self.fig[idx].ylabel('empirical FC', labelpad=-0.5, size=6)
        self.fig[idx].ax.set_ylabel('Empirical FC', labelpad=-0.5, size=6, position=(0.0,0.5))
        self.fig[idx].ax.yaxis.set_label_coords(-0.22,0.5)

        self.fig[(idx, 1)].no_ticks(yaxis=False, nolabels=True)
        self.fig[(idx, 2)].no_ticks(xaxis=False, nolabels=True)
        self.fig[(idx, 1)].remove_y_axis()
        #self.fig[(idx, 1)].remove_x_axis()
        #self.fig[(idx, 1)].xtick_count(2)
        self.fig[(idx, 2)].remove_x_axis()
        #self.fig[(idx, 2)].remove_y_axis()
        #self.fig[(idx, 2)].ytick_count(2)

        #xlims1 = self.fig[(idx, 1)].ax.xaxis.axes.get_xlim()
        #ylims2 = self.fig[(idx, 2)].ax.yaxis.axes.get_ylim()

        #x_labels1 = ['{:0.0f}'.format(xlims1[0]), '{:3.2f}'.format(xlims1[1])]
        #y_labels2 = ['{:0.0f}'.format(ylims2[0]), '{:3.2f}'.format(ylims2[1])]
        #self.fig[(idx, 1)].ax.set_xticklabels(x_labels1, minor=False)
        #self.fig[(idx, 2)].ax.set_yticklabels(y_labels2, minor=False)

        self.fig[(idx, 1)].xlabel('PDF', size=6)
        self.fig[(idx, 2)].ylabel('PDF', size=6)

        self.fig[idx].detach()

    def plot_scatter_dist_sc(self, idx, x, y, cat, kde=True):
        if cat == 'homogeneous':
            clr = self.clr_hom
        elif cat == 'heterogeneous':
            clr = self.clr_het
        else:
            clr = self.clr_emp/3.0

        self.fig[idx].reg_plot(y, x,
                        colors_scatter=clr, colors_line=[0.0, 0.0, 0.0], use_spearman=True, text_fw='normal',
                        text_fs=6, text_loc=(0.5, 1.0), mec=clr, mew=0.0, ms=1.2, lw=0.8)

        self.fig.add_subaxis(idx, 'top', '22%', 0.1, sharex=True)

        ##self.fig[(idx, 1)].distplot([y], np.linspace(-0.2, 1.0, 50), colors=self.clr_emp, kde=True,
        #                               ho=True, lw=0.5, alpha=0.8)
        self.fig[(idx, 1)].distplot([x], np.linspace(-0.05, 1.0, 300), colors=clr, kde=True, lw=0.5, alpha=0.8)
        self.fig[(idx, 1)].no_ticks(xaxis=False, nolabels=True)
        self.fig[(idx, 1)].remove_x_axis()


        self.fig[idx].ax.set_ylim([-0.2, 1.0])
        self.fig[idx].ax.set_xlim([0.0, 1.0])
        self.fig[idx].ax.set_yticks([-0.2, 0.2, 0.6, 1.0])
        self.fig[idx].ax.set_xticks([0.0, 0.5, 1.0])
        #self.fig[idx].xlabel('Structural Connectivity (Conn-1)', size=6)
        #self.fig[idx].ax.set_ylabel('Functional Connectivity', labelpad=-0.5, size=6, position=(0.0, 0.5))
        self.fig[idx].detach()


    def plot_scatter_dist_sc_log(self, idx, x, y, cat, kde=True):
        if cat == 'homogeneous':
            clr = self.clr_hom
        elif cat == 'heterogeneous':
            clr = self.clr_het
        else:
            clr = self.clr_emp/3.0

        self.fig[idx].reg_plot(y, x,
                        colors_scatter=clr, colors_line=[0.0, 0.0, 0.0], use_spearman=True, text_fw='normal',
                        text_fs=6, text_loc=(0.5, 1.0), mec=clr, mew=0.0, ms=1.2, lw=0.8)

        self.fig.add_subaxis(idx, 'top', '22%', 0.1, sharex=True)

        ##self.fig[(idx, 1)].distplot([y], np.linspace(-0.2, 1.0, 50), colors=self.clr_emp, kde=True,
        #                               ho=True, lw=0.5, alpha=0.8)
        self.fig[(idx, 1)].distplot([x], np.linspace(-18.0, 0.0, 300), colors=clr, kde=True, lw=0.5, alpha=0.8)
        self.fig[(idx, 1)].no_ticks(xaxis=False, nolabels=True)
        self.fig[(idx, 1)].remove_x_axis()


        self.fig[idx].ax.set_ylim([-0.2, 1.0])
        self.fig[idx].ax.set_xlim([-18.0, 0.0])
        self.fig[idx].ax.set_yticks([-0.2, 0.2, 0.6, 1.0])
        self.fig[idx].ax.set_xticks([-18.0, 0.0])
        #self.fig[idx].xlabel('Structural Connectivity (Conn-1)', size=6)
        #self.fig[idx].ax.set_ylabel('Functional Connectivity', labelpad=-0.5, size=6, position=(0.0, 0.5))
        self.fig[idx].detach()


    def plot_hist_multi(self, idx, sc, fc_ind_het, fc_ind_hom, sc_av, fc_av_het, fc_av_hom):
        from scipy.stats import gaussian_kde
        density_sc = gaussian_kde(sc)#, bw_method=0.25)
        density_fc_hom = gaussian_kde(fc_ind_hom)#, bw_method=0.25)
        density_fc_het = gaussian_kde(fc_ind_het)  # , bw_method=0.25)

        xx = np.linspace(0.1, 0.7, 100)
        pdf_sc = density_sc(xx); pdf_sc = pdf_sc / pdf_sc.sum()
        pdf_fc_hom = density_fc_hom(xx); pdf_fc_hom = pdf_fc_hom / pdf_fc_hom.sum()
        pdf_fc_het = density_fc_het(xx); pdf_fc_het = pdf_fc_het / pdf_fc_het.sum()

        self.fig[idx].ax.fill(xx, pdf_sc, color=self.clr_emp, fc=self.clr_emp, alpha=0.5)
        self.fig[idx].ax.fill(xx, pdf_fc_hom, color=self.clr_hom, fc=self.clr_hom, alpha=0.5)
        self.fig[idx].ax.fill(xx, pdf_fc_het, color=self.clr_het, fc=self.clr_het, alpha=0.5)

        self.fig[idx].plot(pdf_sc, xx, colors=self.clr_emp, alpha=0.1)
        self.fig[idx].plot(pdf_fc_hom, xx, colors=self.clr_hom, alpha=0.1)
        self.fig[idx].plot(pdf_fc_het, xx, colors=self.clr_het, alpha=0.1)

        self.fig[idx].hline(fc_av_hom, color=self.clr_hom)
        self.fig[idx].hline(fc_av_het, color=self.clr_het)
        self.fig[idx].hline(sc_av, color=self.clr_emp)

        self.fig[idx].ylabel('pdf', size=6)
        self.fig[idx].xlabel('correlation coefficient (r)', size=6)
        self.fig[idx].ax.text(0.5, 1.1, 'individual subject fit distribution', fontsize=6, fontweight='normal',
                        verticalalignment='center', horizontalalignment='center', transform=self.fig[idx].ax.transAxes)


    def plot_scatter(self, idx, x, y, cat, type = 'rank'):
        if cat == 'homogeneous':
            clr = self.clr_hom
        elif cat == 'heterogeneous':
            clr = self.clr_het
        else:
            clr = self.clr_emp

        if type == 'rank':
            r = spearmanr(x, y)
        else:
            r = pearsonr(x, y)

        self.fig[idx].plot(y, x, scatter=True, colors=clr, mew=0.0, ms=1.5, alpha=0.5)


        #self.fig[idx].xtick_count(2)
        #self.fig[idx].ytick_count(2)
        self.fig[idx].set_yticks([y.min(), y.max()])
        self.fig[idx].set_xticks([x.min(), x.max()])
        #self.fig[idx].set_xtick_format(f = '%3.1f')
        #self.fig[idx].set_ytick_format(f = '%3.2f')
        self.fig[idx].xlabel('Empirical', size=6)
        self.fig[idx].ylabel('Model', size=6)

        self.fig[idx].detach()

        if r[1] < 0.001:
            p_text = '***'
        elif r[1] < 0.01:
            p_text = '**'
        elif r[1] < 0.05:
            p_text = '*'
        else:
            p_text = ''

        if type == 'rank':
            title_reg = r'$r_{s} =$' + ' {:3.3f}'.format(r[0]) + '' + p_text
        else:
            title_reg = r'$r_{p} =$' + ' {:3.3f}'.format(r[0]) + '' + p_text

        self.fig[idx].title(title_reg, size=6)


    def plot_phase_diagram_full(self, idx, JEI, Jrec):
        color_list = [[0, 0, 0], [0.1, 0.6, 0.2],
                      [0., 0., 0.], [0.1, 0.6, 0.2]]
        line_list = ['-', '-', '-', '-']

        self.fig[idx].plot(JEI, Jrec, colors=color_list, line_list=line_list, lw=0.8, alpha=1.0)
        self.fig[idx].add_legend(['Critical point', 'Transition to oscillations'], tight=False, anchor=(1.0, 0.2))


        #self.fig[idx].ax.text(1.5, 6.0, 'stable node')
        #self.fig[idx].ax.text(0.25, 12.0, 'unstable node')

        #self.fig[idx].ax.text(1.5, 10.0, 'stable spiral', color=[0.9,0.9,0.9], rotation=15)
        #self.fig[idx].ax.text(1.4, 14.0, 'unstable spiral', color=[0.9,0.9,0.9], rotation=30)

        self.fig[idx].xlabel('$w^{EI}$')
        self.fig[idx].ylabel('$w^{EE}$')
        self.fig[idx].title('Phase Diagram (' + r'$I_{ext}$' + ' = 0)')
        self.fig[idx].detach()


    def plot_phase_diagram(self, idx, JEI, Jrec):
        JEIro = [JEI[ii] for ii in [0, 2, 1, 3]]
        Jrecro = [Jrec[ii] for ii in [0, 2, 1, 3]]
        JEI = JEIro
        Jrec = Jrecro

        color_list = [[0, 0, 0], [0.1, 0.6, 0.2],
                      [0., 0., 0.], [0.1, 0.6, 0.2]]
        line_list = ['-', '-', '-', '-']

        self.fig[idx].plot(JEI, Jrec, colors=color_list, line_list=line_list, lw=1.0, alpha=1.0)
        self.fig[idx].add_legend(['Critical point', 'Transition to oscillations'], tight=False, anchor=(1.0, 0.2))

        self.fig[idx].ax.text(1.75, 5.5, 'Stable \n (asynchronous)', verticalalignment='center', horizontalalignment='center')
        self.fig[idx].ax.text(0.35, 9.5, 'Unstable')

        self.fig[idx].ax.text(1.75, 10.0, r'\textbf{Stable}' + '\n' + r'\textbf{(oscillatory)}', color=[0.9,0.9,0.9], rotation=15, verticalalignment='center', horizontalalignment='center')
        self.fig[idx].ax.text(1.55, 13.0, r'\textbf{Unstable}', color=[0.9,0.9,0.9], rotation=15)

        #self.fig[idx].ax.set_xlim([0.0, 2.25])
        self.fig[idx].ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.25], minor=False)
        self.fig[idx].ax.set_xticklabels([0.0, 0.5, 1.0, 1.5, 2.0, ''])
        #

        self.fig[idx].xlabel('$w^{EI}$')
        self.fig[idx].ylabel('$w^{EE}$')
        self.fig[idx].title('Phase diagram (' + r'$I_{ext}$' + ' = 0 nA)')
        self.fig[idx].detach()

    def plot_phase_diagram_model(self, idx, JEI, Jrec, jee, jei, model, I_ext):
        JEIro = [JEI[ii] for ii in [0, 2, 1, 3]]
        Jrecro = [Jrec[ii] for ii in [0, 2, 1, 3]]
        JEI = JEIro
        Jrec = Jrecro

        color_list = [[0, 0, 0], [0.1, 0.6, 0.2],
                      [0., 0., 0.], [0.1, 0.6, 0.2]]
        line_list = ['-', '-', '-', '-']
        self.fig[idx].plot(JEI, Jrec, colors=color_list, line_list=line_list, lw=0.8)
        if model == 'Heterogeneous':
            #
            ind = jei.argsort()

            #idx1 = jei < 1.15
            #idx2 = np.logical_and(jei >= 1.1, jei < 1.3)
            #idx3 = np.logical_and(jei >= 1.25, jei < 1.5)
            #import pdb;
            #pdb.set_trace()
            for ii in range(len(ind)-1):
                self.fig[idx].plot(jee[ind[ii:]], jei[ind[ii:]], scatter=False, colors=self.clr_het/((ii/180.0+1.0)), lw=1.8,
                               alpha=0.5)  # , sem=False)
            self.fig[idx].ax.text(0.95, 3.1, 'Sens.', rotation=35)
            self.fig[idx].ax.text(1.2, 7.5, 'Assoc.', rotation=35)

            #self.fig[idx].plot(jee[idx1], jei[idx1], scatter=False, colors=self.clr_het, lw=0.5, alpha=0.7)  # , sem=False)
            #self.fig[idx].plot(jee[idx2], jei[idx2], scatter=False, colors=self.clr_het/1.3, lw=0.5, alpha=0.7)  # , sem=False)
            #self.fig[idx].plot(jee[idx3], jei[idx3], scatter=False, colors=self.clr_het/2.0, lw=0.5, alpha=0.7)  # , sem=False)
        else:
            self.fig[idx].plot([jee], [jei], scatter=True, colors=self.clr_hom, lw=0.5, alpha=1.0)  # , sem=False)

        self.fig[idx].xlabel('$w^{EI}$')
        self.fig[idx].ylabel('$w^{EE}$')
        self.fig[idx].ax.set_ylim([0, 14])
        self.fig[idx].set_yticks([0,14], f = '{:3.0f}')

        #self.fig[idx].ytick_count(2)
        self.fig[idx].xtick_count(2)
        self.fig[idx].detach()

        self.fig[idx].title(model + '\n model \n (' + r'$I_{ext}$' + ' = {:3.3f} nA)'.format(I_ext))

    def plot_parameter_space(self, idx, JEI, Jrec, theta_hom, theta_het, myelin_gradient):


        jei_hom = theta_hom[0]
        jee_hom = theta_hom[1]

        #idx = myelin_gradient.argsort()
        jei = np.array([theta_het[0][ii] + theta_het[1][ii] * myelin_gradient for ii in range(25)])
        jee = np.array([theta_het[2][ii] + theta_het[3][ii] * myelin_gradient for ii in range(25)])

        bnd1 = jei.min()
        bnd2 = jei.max()

        jei_het = jei.mean(0)
        jee_het = jee.mean(0)

        color_list = [[0, 0, 0], [0.0, 0.0, 0.0],
                      [0, 0, 0]]
        line_list = ['-', '-', ':', '-']

        #import pdb; pdb.set_trace()

        #self.fig[idx].add_legend(['critical (pitchfork)', 'critical (hopf)'], tight=False, anchor=(1.05, 0.4))

        JEI2 = []
        Jrec2 = []
        for ii in range(len(JEI)-1):
            ix1 = np.abs(np.array(Jrec[ii]) - bnd1).argmin()
            ix2 = np.abs(np.array(Jrec[ii]) - bnd2).argmin()
            Jrec2 += [Jrec[ii][ix1:ix2+1]]
            JEI2 += [(np.array(JEI[ii][ix1:ix2+1]) + 0.35).tolist()]

        #import pdb; pdb.set_trace()

        self.fig[idx].plot(JEI2, Jrec2, colors=color_list, line_list=line_list, lw=1.0)


        self.fig[idx].ax.fill_between(Jrec2[2], JEI2[2], JEI2[1], color=[0.5, 0.5, 0.5], alpha=0.3)

        clr = np.tile(self.clr_het, (jee.shape[0],1))
        self.fig[idx].plot(jee, jei, scatter=True, colors=clr, mec=clr[0], mew=0.0, ms=1.2,  lw=0.4, alpha=0.3)#, sem=False)
        self.fig[idx].plot(jee_hom, jei_hom, scatter=True, colors=self.clr_hom, mec=self.clr_hom, lw=0.4, alpha=0.3)  # , sem=False)

        '''
        import scipy.stats as st
        #import matplotlib.pyplot as pl
        # xmin, xmax = 1.25, 1.45
        # ymin, ymax = 9.0, 10.5
        #xmin, xmax = 0.5, 5.45
        #ymin, ymax = 9.0, 10.5
        xmin, xmax = bnd1, bnd2
        ymin, ymax = jee.min(), jee.max()

        theta_het2 = np.vstack((jei.ravel(), jee.ravel()))

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        #values = theta_hom[:2, :]
        values = theta_het2
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        #import pdb;
        #pdb.set_trace()
        f = np.ma.array(f)
        f[f<0.5] = np.ma.masked


        #ax.set_xlim(xmin, xmax)
        #ax.set_ylim(ymin, ymax)
        # Contourf plot
        self.fig[idx].ax.contourf(xx, yy, f, cmap='OrRd')
        ## Or kernel density estimate plot instead of the contourf plot
        # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        # Contour plot
        # cset = ax.contour(xx, yy, f, colors='k')
        #pl.show()
        '''

        #self.fig[idx].plot(jee_het, jei_het, colors=self.clr_het, lw=1.0)

        #import pdb; pdb.set_trace()


    def plot_bifurcation(self, idx, pars, points, markers):
        S_E_ss = 0.1647572075
        se = pars[0]
        se_unstable = pars[1]
        jn = pars[2]
        cp_val = pars[3]
        cp = pars[4]

        tick_max = 1.0
        marker_pad = 0.05
        alignment = ['right', 'left']

        cp_up = np.where(np.abs(se[0, :] - S_E_ss) > 0.0001)[0][0]
        cp_down = np.where(np.abs(se[1, :] - S_E_ss) > 0.0001)[0][0]

        self.fig[idx].plot([S_E_ss, S_E_ss], [0, cp_val], colors=[0, 0, 0])
        self.fig[idx].plot([S_E_ss, S_E_ss], [cp_val, jn[-1]], colors=[0, 0, 0], line_list=[':'])

        if cp_up < cp:
            self.fig[idx].plot(se_unstable[0, cp_up:cp], jn[cp_up:cp], colors=[0, 0, 0], line_list=[':'])

        if cp_down < cp:
            self.fig[idx].plot(se_unstable[1, cp_down:cp], jn[cp_down:cp], colors=[0, 0, 0], line_list=[':'])

        self.fig[idx].plot(se[0, cp_up:], jn[cp_up:], colors=[0, 0, 0])
        self.fig[idx].plot(se[1, cp_down:], jn[cp_down:], colors=[0, 0, 0])

        self.fig[idx].plot([S_E_ss, S_E_ss], points, scatter=True, colors=[0, 0, 0.1], ms=2.5, mew=1.0)

        self.fig[idx].ax.text(points[0], S_E_ss + marker_pad, markers[0], fontsize=6,
                               verticalalignment='bottom',
                               horizontalalignment=alignment[0])

        self.fig[idx].ax.text(points[1], S_E_ss + marker_pad, markers[1], fontsize=6,
                               verticalalignment='bottom',
                               horizontalalignment=alignment[1])

        self.fig[idx].detach()
        self.fig[idx].set_yticks(np.linspace(0.0, tick_max, 4))
        self.fig[idx].ylabel('$S^{E}$')
        self.fig[idx].xlabel('$w^{EE}$')


    def plot_perturbations(self, idx, pars, sim_times, formats):
        SE_up = pars[0]
        SE_down = pars[1]

        color = [0, 0, 0]
        self.fig[idx].plot(SE_up, np.arange(sim_times) / 10, colors=color)
        self.fig[idx].plot(SE_down, np.arange(sim_times) / 10, colors=[0.5, 0.5, 0.5])
        self.fig[idx].set_xticks(np.linspace(0, sim_times / 10, 5), f='{:3.0f}')
        ylims = self.fig[idx].ax.get_ylim()
        self.fig[idx].set_yticks(np.linspace(ylims[0], ylims[1], 3), f=formats)
        self.fig[idx].detach()
        self.fig[idx].xtick_count(3)
        self.fig[idx].ylabel('$S^{E}$')
        self.fig[idx].xlabel('time (ms)')


    def network_bar3(self, idx, r_net_het, r_net_hom,
                    legend=False, anchor=[1.1,1.1],
                    xlims=[0.5, 9.5], ylims=[0.0, 1.0],
                    ytc = 6,
                    ylabel = 'pearson-r'):
        x_axis = [1, 2, 3, 4.5, 5.5, 6.5, 7.5, 8.5]

        self.fig[idx].bar([r_net_hom.tolist(), r_net_het.tolist()], x=[x_axis, x_axis], colors=[self.clr_hom, self.clr_het],
                   x_labels=self.network_labels, lw=0.5, alpha=0.9)
        self.fig[idx].ax.set_ylim(ylims)
        self.fig[idx].ax.set_xlim(xlims)
        self.fig[idx].ytick_count(ytc)
        self.fig[idx].ylabel(ylabel)
        if legend:
            self.fig[idx].add_legend(['Heterogeneous', 'Homogeneous'], anchor=anchor, handlelength=1.5)

    def network_bar2(self, idx, r_net_het, r_net_hom, r_sys, sensory_ix, association_ix,
                    legend=False, anchor=[1.1,1.1],
                    xlims=[0.5, 9.5], ylims=[0.0, 1.0],
                    ytc = 6,
                    ylabel = 'Correlation'):

        self.fig.add_subaxis(idx, "right", "30%", pad=0.05, sharey=True)

        x_axis = [1, 2, 3, 4.5, 5.5, 6.5, 7.5, 8.5]
        d = .015
        clr = [self.clr_emp, self.clr_emp, self.clr_emp,
               self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0]
        #hatchs = ('','','','\\','\\','\\','\\','\\')

        self.fig[idx, 0].bar([r_net_hom.tolist(), r_net_het.tolist()], x=[x_axis, x_axis], colors=[self.clr_hom, self.clr_het],
                   x_labels=self.network_labels, lw=0.5, alpha=0.9)
        self.fig[idx, 0].ax.set_ylim(ylims)
        self.fig[idx, 0].ax.set_xlim(xlims)
        self.fig[idx, 0].ytick_count(ytc)
        self.fig[idx, 0].ylabel(ylabel)

        self.fig[idx, 1].bar(r_sys,
                            colors=[self.clr_hom, self.clr_het],
                            x_labels=['Sensory', 'Association'], width=0.75, lw=0.5, alpha=0.8)
        # fig03.fig[0, 1].boxplot([myelin_sys[0], myelin_sys[1]],
        #                    colors=[fig03.clr_emp, fig03.clr_emp],
        #                    x_labels=['sensory', 'association'], width=0.55, lw=0.5, alpha=0.5)
        self.fig[idx, 1].remove_y_axis()
        self.fig[idx, 1].ax.spines['left'].set_visible(False)
        self.fig[idx, 1].ax.set_xlim([0.25, 2.75])

        for tick in self.fig[idx, 1].ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_horizontalalignment("right")

        self.fig[idx, 0].ax.plot((1 - d / 3., 1 + d / 3.), (-d, +d), transform=self.fig[idx, 0].ax.transAxes,
                                color=self.fig[idx, 0].almost_black, lw=0.5, clip_on=False)
        self.fig[idx, 1].ax.plot((-d, +d), (-d, +d), transform=self.fig[idx, 1].ax.transAxes,
                                color=self.fig[idx, 0].almost_black, lw=0.5, clip_on=False, label='_nolegend_')

        self.fig[idx, 0].ax.plot([x_axis[0], x_axis[2]], [sensory_ix[1], sensory_ix[1]],
                                color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([x_axis[0], x_axis[0]], [sensory_ix[0], sensory_ix[1]],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([x_axis[2], x_axis[2]], [sensory_ix[0], sensory_ix[1]],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)

        self.fig[idx, 0].ax.text((x_axis[0]+x_axis[2])/2.0, sensory_ix[1]+0.075, 'Sensory', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

        self.fig[idx, 0].ax.plot([x_axis[3], x_axis[7]], [association_ix[1], association_ix[1]],
                                color=self.fig[idx, 0].almost_black, lw=0.7, clip_on=False)
        self.fig[idx, 0].ax.plot([x_axis[3], x_axis[3]], [association_ix[0], association_ix[1]],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([x_axis[7], x_axis[7]], [association_ix[0], association_ix[1]],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)

        self.fig[idx, 0].ax.text((x_axis[3]+x_axis[7])/2.0, association_ix[1] + 0.075, 'Association', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

    def network_bar(self, idx, r, r_sys):
        self.fig.add_subaxis(idx, "right", "30%", pad=0.05, sharey=True)

        x_axis = [1, 2, 3, 4.5, 5.5, 6.5, 7.5, 8.5]
        d = .015
        network_colors = [[41, 252, 252, 255], [74, 145, 242, 255], [41, 142, 143, 255],
                          [245, 155, 95, 255], [255, 233, 175, 255], [246, 102, 41, 255],
                          [231, 91, 115, 255], [212, 172, 244, 255]]
        network_colors = (1.0 * np.array(network_colors)) / 255.0
        clr = [self.clr_emp, self.clr_emp, self.clr_emp,
               self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0]
        #hatchs = ('','','','\\','\\','\\','\\','\\')

        self.fig[idx, 0].bar(r, x=x_axis, colors=network_colors, x_labels=self.network_labels, width=0.75,
                            lw=0.5,
                            alpha=0.8)

        self.fig[idx, 0].ax.set_ylim([1.2, 1.5])
        self.fig[idx, 0].ax.set_xlim([0.5, 9.5])
        #self.fig[idx, 0].no_ticks(yaxis=False)

        self.fig[idx, 0].ylabel('T1w/T2w value')

        self.fig[idx, 0].ax.text(0.7, 1.1, 'Mean T1w/T2w value per network', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center',
                                transform=self.fig[idx, 0].ax.transAxes)

        self.fig[idx, 1].bar([r_sys[0], r_sys[1]],
                            colors=[self.clr_emp, self.clr_emp/1.0],
                            x_labels=['Sensory', 'Association'], width=0.75, lw=0.5, alpha=0.8)
        # fig03.fig[0, 1].boxplot([myelin_sys[0], myelin_sys[1]],
        #                    colors=[fig03.clr_emp, fig03.clr_emp],
        #                    x_labels=['sensory', 'association'], width=0.55, lw=0.5, alpha=0.5)
        self.fig[idx, 1].remove_y_axis()
        self.fig[idx, 1].ax.spines['left'].set_visible(False)
        self.fig[idx, 1].ax.set_xlim([0.5, 3.0])
        self.fig[idx, 1].ax.plot([1.25, 2.25], [1.43, 1.43], 'k', lw=0.7)
        self.fig[idx, 1].ax.text(1.75, 1.43, r'\textbf{***}', color=[0.0, 0.0, 0.0], fontsize=7,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

        for tick in self.fig[idx, 0].ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_horizontalalignment("right")

        for tick in self.fig[idx, 1].ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_horizontalalignment("right")

        self.fig[idx, 0].ax.plot((1 - d / 3., 1 + d / 3.), (-d, +d), transform=self.fig[idx, 0].ax.transAxes,
                                color=self.fig[idx, 0].almost_black, lw=0.5, clip_on=False)
        self.fig[idx, 1].ax.plot((-d, +d), (-d, +d), transform=self.fig[idx, 1].ax.transAxes,
                                color=self.fig[idx, 0].almost_black, lw=0.5, clip_on=False)

        self.fig[idx, 0].ax.plot([1.25, 3.25], [1.5, 1.5],
                                color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([1.25, 1.25], [1.49, 1.5],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([3.25, 3.25], [1.49, 1.5],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)

        self.fig[idx, 0].ax.text(2.25, 1.51, 'Sensory', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

        self.fig[idx, 0].ax.plot([5., 9.], [1.32, 1.32],
                                color=self.fig[idx, 0].almost_black, lw=0.7, clip_on=False)
        self.fig[idx, 0].ax.plot([5., 5.], [1.31, 1.32],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([9., 9.], [1.31, 1.32],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)

        self.fig[idx, 0].ax.text(6.75, 1.33, 'Association', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

    def network_bar_witherr(self, idx, r, r_sys):
        self.fig.add_subaxis(idx, "right", "30%", pad=0.05, sharey=True)

        x_axis = [1, 2, 3, 4.5, 5.5, 6.5, 7.5, 8.5]
        d = .015
        network_colors = [[41, 252, 252, 255], [74, 145, 242, 255], [41, 142, 143, 255],
                          [245, 155, 95, 255], [255, 233, 175, 255], [246, 102, 41, 255],
                          [231, 91, 115, 255], [212, 172, 244, 255]]
        network_colors = (1.0 * np.array(network_colors)) / 255.0
        clr = [self.clr_emp, self.clr_emp, self.clr_emp,
               self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0, self.clr_emp/1.0]
        #hatchs = ('','','','\\','\\','\\','\\','\\')

        self.fig[idx, 0].bar(r, x=x_axis, colors=network_colors, x_labels=self.network_labels, width=0.75,
                            lw=0.5,
                            alpha=0.8)

        self.fig[idx, 0].ax.set_ylim([1.1, 1.7])
        self.fig[idx, 0].ax.set_xlim([0.5, 9.5])
        #self.fig[idx, 0].no_ticks(yaxis=False)

        self.fig[idx, 0].ylabel('T1w/T2w value')

        self.fig[idx, 0].ax.text(0.7, 1.1, 'Mean T1w/T2w value per network', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center',
                                transform=self.fig[idx, 0].ax.transAxes)

        self.fig[idx, 1].bar([r_sys[0], r_sys[1]],
                            colors=[self.clr_emp, self.clr_emp/1.0],
                            x_labels=['Sensory', 'Association'], width=0.75, lw=0.5, alpha=0.8)
        # fig03.fig[0, 1].boxplot([myelin_sys[0], myelin_sys[1]],
        #                    colors=[fig03.clr_emp, fig03.clr_emp],
        #                    x_labels=['sensory', 'association'], width=0.55, lw=0.5, alpha=0.5)
        self.fig[idx, 1].remove_y_axis()
        self.fig[idx, 1].ax.spines['left'].set_visible(False)
        self.fig[idx, 1].ax.set_xlim([0.5, 3.0])
        self.fig[idx, 1].ax.plot([1.25, 2.25], [1.43, 1.43], 'k', lw=0.7)
        self.fig[idx, 1].ax.text(1.75, 1.43, r'\textbf{***}', color=[0.0, 0.0, 0.0], fontsize=7,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

        for tick in self.fig[idx, 0].ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_horizontalalignment("right")

        for tick in self.fig[idx, 1].ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_horizontalalignment("right")

        self.fig[idx, 0].ax.plot((1 - d / 3., 1 + d / 3.), (-d, +d), transform=self.fig[idx, 0].ax.transAxes,
                                color=self.fig[idx, 0].almost_black, lw=0.5, clip_on=False)
        self.fig[idx, 1].ax.plot((-d, +d), (-d, +d), transform=self.fig[idx, 1].ax.transAxes,
                                color=self.fig[idx, 0].almost_black, lw=0.5, clip_on=False)

        self.fig[idx, 0].ax.plot([1.25, 3.25], [1.65, 1.65],
                                color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([1.25, 1.25], [1.64, 1.65],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([3.25, 3.25], [1.64, 1.65],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)

        self.fig[idx, 0].ax.text(2.25, 1.67, 'Sensory', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')

        self.fig[idx, 0].ax.plot([5., 9.], [1.4, 1.4],
                                color=self.fig[idx, 0].almost_black, lw=0.7, clip_on=False)
        self.fig[idx, 0].ax.plot([5., 5.], [1.39, 1.4],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)
        self.fig[idx, 0].ax.plot([9., 9.], [1.39, 1.4],
                                 color=self.fig[idx, 0].almost_black, lw=0.8, clip_on=False)

        self.fig[idx, 0].ax.text(6.75, 1.42, 'Association', color=[0.0, 0.0, 0.0], fontsize=6,
                                fontweight='normal',
                                verticalalignment='center',
                                horizontalalignment='center')


    def plot_psd(self, idx, x, y, cat, labels, myelin, bands=None, loc='bottom', iscbar=True):
        lw = 0.9
        alpha = 0.7
        if cat == 'homogeneous':
            if y.ndim > 1:
                if y.shape[0] > 4:
                    clr = np.tile(self.clr_hom, (360, 1))
                    for ii in range(360):
                        clr[ii,:] = (1.0-myelin[ii]*0.75)*clr[ii,:]
                    #clr = self.clr_hom
                    lw = 0.3
                    alpha = 0.3
                else:
                    clr = [self.clr_hom, self.clr_hom/1.5]
                    #clr = self.clr_hom
            else:
                clr = self.clr_hom
        elif cat == 'heterogeneous':
            if y.ndim > 1:
                if y.shape[0] > 4:
                    clr = np.tile(self.clr_het, (360, 1))
                    for ii in range(360):
                        clr[ii,:] = (1.0-myelin[ii]*0.75)*clr[ii,:]
                    lw = 0.3
                    alpha = 0.3
                else:
                    clr = [self.clr_het, self.clr_het/1.5]
            else:
                clr = self.clr_het
        else:
            clr = np.tile(self.clr_emp*1.3, (360, 1))
            for ii in range(360):
                clr[ii, :] = (1.0 - myelin[ii]) * clr[ii, :]
            lw = 0.3
            alpha = 0.3

        if x[0] == 0:
            x[0] = 0.1

        self.fig[idx].plot(10*np.log10(y), np.log10(x), colors=clr, lw=lw, alpha=alpha)

        self.fig[idx].xlabel('Frequency (Hz)')
        self.fig[idx].ylabel('Power (dB)')
        self.fig[idx].ax.set_xticks(np.log10(labels))

        if labels[0] == 0.005:
            x_labels1 = ['{:0.0f}'.format(labels[ii]) for ii in range(len(labels))]
            x_labels1[0] = '{:3.3f}'.format(labels[0])
        else:
            x_labels1 = ['{:0.0f}'.format(labels[ii]) for ii in range(len(labels))]
            if cat is not 'empirical':
                x_labels1[0] =r'$<$'+'{:3.1f}'.format(labels[0])

        from matplotlib.colors import LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list('SenAss', clr[myelin.argsort(),:], N=360)
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=180)
        if loc=='right':
            self.fig[idx].append_cbar(cm, norm=norm, cbarpad = 0.05, cbarloc='right',shrinked=True, cbartick=2)
            self.fig[idx].cb.ax.set_yticklabels(['Sensory', 'Association'])
        else:
            if cat is not 'empirical':
                self.fig[idx].append_cbar(cm, norm=norm, cbarpad=0.26, cbarloc='bottom', shrinked=False, cbartick=2,
                                          cbar_title='Heterogeneity map', labelpad=-5)
                self.fig[idx].cb.ax.text(-0.03, 0.075, '0', fontsize=6, fontweight='normal',
                                     verticalalignment='center',
                                     horizontalalignment='center',
                                      transform=self.fig[idx].cb.ax.transAxes)
                self.fig[idx].cb.ax.text(1.03, 0.075, '1', fontsize=6, fontweight='normal',
                                         verticalalignment='center',
                                         horizontalalignment='center',
                                         transform=self.fig[idx].cb.ax.transAxes)

                #self.fig[idx].cb.ax.set_position([0.05, 0.3, 0.5, 0.4])


                self.fig[idx].cb.ax.set_xticklabels(['Sensory', 'Association'])
            else:
                self.fig[idx].append_cbar(cm, norm=norm, cbarpad=0.26, cbarloc='bottom', anchor=(1.5, 0.0), shrinked=False, cbartick=2,
                                          cbar_title='Heterogeneity map', labelpad=-5)
                self.fig[idx].cb.ax.text(-0.03, 0.075, '0', fontsize=6, fontweight='normal',
                                         verticalalignment='center',
                                         horizontalalignment='center',
                                         transform=self.fig[idx].cb.ax.transAxes)
                self.fig[idx].cb.ax.text(1.03, 0.075, '1', fontsize=6, fontweight='normal',
                                         verticalalignment='center',
                                         horizontalalignment='center',
                                         transform=self.fig[idx].cb.ax.transAxes)

                # self.fig[idx].cb.ax.set_position([0.05, 0.3, 0.5, 0.4])


                self.fig[idx].cb.ax.set_xticklabels(['Sensory', 'Association'])

        self.fig[idx].ax.set_xticklabels(x_labels1)
        self.fig[idx].ytick_count(5)
        self.fig[idx].detach()

        if bands is not None:
            band_labels = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']

            self.fig[idx].ax.text(np.log10(0.5 * (bands[0][0] + bands[0][1])), -6, band_labels[0], fontsize=8,
                                 verticalalignment='center', horizontalalignment='center')
            self.fig[idx].hline(np.log10(4), color=[0., 0., 0.], ls=':')

            self.fig[idx].ax.text(np.log10(0.5 * (bands[1][0] + bands[1][1])), -6, band_labels[1], fontsize=8,
                                 verticalalignment='center', horizontalalignment='center')
            self.fig[idx].hline(np.log10(8), color=[0., 0., 0.], ls=':')

            self.fig[idx].ax.text(np.log10(0.5 * (bands[2][0] + bands[2][1])), -6, band_labels[2], fontsize=8,
                                 verticalalignment='center', horizontalalignment='center')
            self.fig[idx].hline(np.log10(16), color=[0., 0., 0.], ls=':')

            self.fig[idx].ax.text(np.log10(0.5 * (bands[3][0] + bands[3][1])), -6, band_labels[3], fontsize=8,
                                 verticalalignment='center', horizontalalignment='center')

            self.fig[idx].hline(np.log10(35), color=[0., 0., 0.], ls=':')

            self.fig[idx].ax.text(np.log10(0.5 * (bands[4][0] + bands[4][1])), -6, band_labels[4], fontsize=8,
                                 verticalalignment='center', horizontalalignment='center')

            self.fig[idx].hline(np.log10(50), color=[0., 0., 0.], ls=':')


    def sig_bar(self, idx, x_locs, vals, sigs, inc_pos = (0.05, 0.07), inc_neg = (-0.026, -0.15)):
        for ii in range(len(vals)):
            if vals[ii] < 0:
                inc = inc_neg[0]
                inc2 = inc_neg[1]
            else:
                inc = inc_pos[0]
                inc2 = inc_pos[1]

            self.fig[idx].ax.plot(x_locs[ii], [vals[ii] + inc, vals[ii] + inc], 'k', lw=0.6)

            if sigs[ii] == 1:
                self.fig[idx].ax.text((x_locs[ii][0] + x_locs[ii][1]) / 2.0, vals[ii] + inc2, r'\textbf{***}',
                                      color=[0.0, 0.0, 0.0], fontsize=6,
                                      fontweight='normal',
                                      verticalalignment='center',
                                      horizontalalignment='center')
            else:
                self.fig[idx].ax.text((x_locs[ii][0] + x_locs[ii][1]) / 2.0, vals[ii] + 0.06, 'n.s.',
                                      color=[0.0, 0.0, 0.0], fontsize=6,
                                      fontweight='normal',
                                      verticalalignment='center',
                                      horizontalalignment='center')



    def violin(self, idx, data, sensory_idx, association_idx, model_clr):
        self.fig[idx].despine()
        self.fig[idx].ax.violinplot([data[sensory_idx], data[association_idx]], [1, 2], points=30, widths=0.5,
                                    showmeans=False, showmedians=False,
                                    showextrema=False)

        quartile1_s, median_s, quartile3_s = np.percentile(data[sensory_idx], [25, 50, 75])
        quartile1_a, median_a, quartile3_a = np.percentile(data[association_idx], [25, 50, 75])

        self.fig[idx].ax.set_ylim([-2.0, 6.0])
        self.fig[idx].ax.collections[0].set_facecolor(model_clr)
        self.fig[idx].ax.collections[1].set_facecolor(model_clr)
        self.fig[idx].ax.collections[0].set_edgecolor(model_clr)
        self.fig[idx].ax.collections[1].set_edgecolor(model_clr)
        self.fig[idx].ax.collections[0].set_alpha(0.8)
        self.fig[idx].ax.collections[1].set_alpha(0.8)
        #for ii in range(2, 6):
        #    self.fig[idx].ax.collections[ii].set_color(self.fig[idx].almost_black)

        self.fig[idx].ax.scatter(1.0, median_s, marker='o', color='white', s=3, zorder=3)
        self.fig[idx].ax.vlines(1.0, quartile1_s, quartile3_s, color='k', linestyle='-', lw=5)
        #self.fig[idx].ax.vlines(1.0, quartile1_s - (quartile3_s - quartile1_s)*1.5, quartile3_s + (quartile3_s - quartile1_s)*1.5, color='k', linestyle='-', lw=1)
        self.fig[idx].ax.vlines(1.0,
                                np.clip(quartile1_s - (quartile3_s - quartile1_s) * 1.5, data[sensory_idx].min(),
                                        quartile1_s),
                                np.clip(quartile3_s + (quartile3_s - quartile1_s) * 1.5, quartile3_s,
                                        data[sensory_idx].max()), color='k', linestyle='-', lw=1)

        self.fig[idx].ax.scatter(2.0, median_a, marker='o', color='white', s=3, zorder=3)
        self.fig[idx].ax.vlines(2.0, quartile1_a, quartile3_a, color='k', linestyle='-', lw=5)
        self.fig[idx].ax.vlines(2.0, np.clip(quartile1_a - (quartile3_a - quartile1_a) * 1.5, data[association_idx].min(), quartile1_a),
                                np.clip(quartile3_a + (quartile3_a - quartile1_a) * 1.5, quartile3_a, data[association_idx].max()), color='k', linestyle='-', lw=1)

        self.fig[idx].ax.set_xticks([1, 2])
        self.fig[idx].ax.set_xticklabels(['Sensory', 'Association'])
        for tick in self.fig[idx].ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_horizontalalignment("right")


    def network_plot2(self, parcel):
        networks = ['Visual', 'Auditory', 'Somatomotor','Dorsal-attention',
                    'Frontoparietal', 'Ventral-attention', 'Default-Mode', 'Cingulo-Opercular',
                     'Hippocampal',
                    'Posterior-Multimodal', 'Unknown-2', 'Unknown-3']
        network_colors = [[74, 145, 242, 255], [41, 252, 252, 255], [41, 142, 143, 255],
                          [245, 155, 95, 255], [255, 233, 175, 255], [246, 102, 41, 255],
                          [231, 91, 115, 255], [212, 172, 244, 255], [100, 100, 100, 255]]
        network_colors = (1.0 * np.array(network_colors)) / 255.0
        from matplotlib import colors

        id_net = []
        clusters = np.zeros(360)
        color_list = np.ones((360, 4))
        color_list[:,:3] *= 0.4
        for ii, nn in enumerate(networks):
            if ii < 8:
                cind = np.where(parcel.network == nn)[0]
                clusters[cind] = ii + 1
                color_list[cind, :] = np.tile(network_colors[ii],(len(cind), 1))
                #import pdb;
                #pdb.set_trace()

            id_net += parcel.index[parcel.network==nn].tolist()

        #clusters = clusters[id_net]
        #color_list = color_list[id_net,:]
        bounds = np.where(np.diff(clusters) != 0.)[0] + 1
        bounds = np.insert(bounds, [0, len(bounds)], [0, 360])

        #import pdb; pdb.set_trace()
        return id_net, clusters, bounds, color_list, networks

    def network_plot(self, parcel):
        networks = ['Visual', 'Auditory', 'Somatomotor', 'Dorsal-attention',
                    'Frontoparietal', 'Ventral-attention', 'Default-Mode', 'Cingulo-Opercular',
                    'Hippocampal',
                    'Posterior-Multimodal', 'Unknown-2', 'Unknown-3']
        network_colors = [[74, 145, 242, 255], [41, 252, 252, 255], [41, 142, 143, 255],
                          [245, 155, 95, 255], [255, 233, 175, 255], [246, 102, 41, 255],
                          [231, 91, 115, 255], [212, 172, 244, 255], [100, 100, 100, 255]]
        network_colors = (1.0 * np.array(network_colors)) / 255.0
        from matplotlib import colors

        id_net = []
        clusters = np.zeros(180)
        color_list = np.ones((180, 4))
        color_list[:, :3] *= 0.4
        for ii, nn in enumerate(networks):
            if ii < 8:
                clusters[parcel.index[parcel.network == nn]] = ii + 1
                color_list[parcel.index[parcel.network == nn], :] = np.tile(network_colors[ii],
                                                                            (
                                                                            len(parcel.index[parcel.network == nn]), 1))
            id_net += parcel.index[parcel.network == nn].tolist()

        clusters = clusters[id_net]
        color_list = color_list[id_net, :]
        bounds = np.where(np.diff(clusters) != 0.)[0] + 1
        bounds = np.insert(bounds, [0, len(bounds)], [0, 180])

        # import pdb; pdb.set_trace()
        return id_net, clusters, bounds, color_list, networks


    def plot_matrix(self, idx, mat, parcel_ordered, cmap='plasma', vmin=0.0, vmax=1.0, cbar_title=None):
        mat = np.copy(mat)


        #data_full.load_parcel()
        #parcel_ordered = data_full.glasser.parcel.sort_values(by=['hemi', 'network'])[::-1]
        #ind = parcel_ordered.index
        #parcel_ordered = parcel_ordered.reset_index()

        # data_full.glasser.sortby(['hemi'])
        # import pdb; pdb.set_trace()
        idx2, clusters, bounds, color_list, networks = self.network_plot(parcel_ordered)

        # idx = data.glasser_L.sort_values(by='network').index[::-1]
        #mat = mat[ind, :][:, ind]

        self.fig[idx].heatmap(mat[idx2,:][:,idx2], cmap=cmap, cbar=True, vmin=vmin, vmax=vmax, cbarloc="bottom",
                              cbartick=2, cbar_title=cbar_title, cbar_title_size=6, cbar_title_weight='normal', labelpad=-0.2, xsize="4%",
                                clusters=bounds, color_list=color_list, labels=networks)
        self.fig[idx].no_ticks()
        self.fig[idx].remove_x_axis()
        self.fig[idx].remove_y_axis()