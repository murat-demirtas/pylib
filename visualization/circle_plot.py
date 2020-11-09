import matplotlib.pyplot as plt
import matplotlib.path as m_path
import matplotlib.patches as m_patches
import numpy as np
import os
from tools import parcels

class Circle():
    def __init__(self, fig, ax, parc='cole', circle_order=None, palette='Set2'):
        self.ax = ax
        self.fig = fig

        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.parent_path = os.path.abspath(os.path.join(self.module_path, os.pardir))
        self.template_path = self.parent_path + '/data/'

        self.parcel = parcels.Parcel(parc=parc)
        self.parcel.sortby(['hemi', 'network'])

        self.labels_onlyname = self.parcel.get_labels(dropna=True)
        self.label_names = self.parcel.get_labels(hemi=True, dropna=True)
        self.label_order = self.parcel.flip_order(self.label_names)
        self.node_colors, self.group_keys, self.bounds = self.parcel.colorby('network')
        self.node_colors = self.node_colors[:,:3]
        self.group_boundaries = [0, len(self.label_names) / 2]
        self.node_angles = self.circular_layout(start_pos=90)


    def circular_layout(self, start_pos=90, start_between=True,
                        group_boundaries=None, group_sep=10):
        """Create layout arranging nodes on a circle.

        Parameters
        ----------
        node_names : list of str
            Node names.
        node_order : list of str
            List with node names defining the order in which the nodes are
            arranged. Must have the elements as node_names but the order can be
            different. The nodes are arranged clockwise starting at "start_pos"
            degrees.
        start_pos : float
            Angle in degrees that defines where the first node is plotted.
        start_between : bool
            If True, the layout starts with the position between the nodes. This is
            the same as adding "180. / len(node_names)" to start_pos.
        group_boundaries : None | array-like
            List of of boundaries between groups at which point a "group_sep" will
            be inserted. E.g. "[0, len(node_names) / 2]" will create two groups.
        group_sep : float
            Group separation angle in degrees. See "group_boundaries".

        Returns
        -------
        node_angles : array, shape=(len(node_names,))
            Node angles in degrees.
        """
        # TODO: incorporate variables into the code
        node_names = self.label_names
        node_order = self.label_order
        group_boundaries = self.group_boundaries
        n_nodes = len(node_names)

        if len(node_order) != n_nodes:
            raise ValueError('node_order has to be the same length as node_names')

        if group_boundaries is not None:
            boundaries = np.array(group_boundaries, dtype=np.int)
            if np.any(boundaries >= n_nodes) or np.any(boundaries < 0):
                raise ValueError('"group_boundaries" has to be between 0 and '
                                 'n_nodes - 1.')
            if len(boundaries) > 1 and np.any(np.diff(boundaries) <= 0):
                raise ValueError('"group_boundaries" must have non-decreasing '
                                 'values.')
            n_group_sep = len(group_boundaries)
        else:
            n_group_sep = 0
            boundaries = None

        # convert it to a list with indices
        node_order = [node_order.index(name) for name in node_names]
        node_order = np.array(node_order)
        if len(np.unique(node_order)) != n_nodes:
            raise ValueError('node_order has repeated entries')

        node_sep = (360. - n_group_sep * group_sep) / n_nodes

        if start_between:
            start_pos += node_sep / 2

            if boundaries is not None and boundaries[0] == 0:
                # special case when a group separator is at the start
                start_pos += group_sep / 2
                boundaries = boundaries[1:] if n_group_sep > 1 else None

        node_angles = np.ones(n_nodes, dtype=np.float) * node_sep
        node_angles[0] = start_pos
        if boundaries is not None:
            node_angles[boundaries] += group_sep

        node_angles = np.cumsum(node_angles)[node_order]

        return node_angles


    def plot(self, con,
             linewidth=0.5, padding=0., node_linewidth=0.05,
             n_lines=None, colormap='autumn', vmin=None, vmax=None,
             names=False, fontsize_names=8, textcolor='black', node_edgecolor='white',
             node_width=None, fontsize_title=12, title=None):


        node_angles = self.circular_layout(start_pos=90)
        node_names = self.labels_onlyname
        con = con[:,self.parcel.order][self.parcel.order,:]

        n_nodes = len(node_names)

        if node_angles is not None:
            if len(node_angles) != n_nodes:
                raise ValueError('node_angles has to be the same length '
                                 'as node_names')
            # convert it to radians
            node_angles = node_angles * np.pi / 180
        else:
            # uniform layout on unit circle
            node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

        if node_width is None:
            # widths correspond to the minimum angle between two nodes
            dist_mat = node_angles[None, :] - node_angles[:, None]
            dist_mat[np.diag_indices(n_nodes)] = 1e9
            node_width = np.min(np.abs(dist_mat))
        else:
            node_width = node_width * np.pi / 180

        node_colors = self.node_colors
        # handle 1D and 2D connectivity information
        if con.ndim == 1:
            if indices is None:
                raise ValueError('indices has to be provided if con.ndim == 1')
        elif con.ndim == 2:
            if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
                raise ValueError('con has to be 1D or a square matrix')
            # we use the lower-triangular part
            indices = np.tril_indices(n_nodes, -1)
            con = con[indices]
        else:
            raise ValueError('con has to be 1D or a square matrix')

        self.cmap = colormap
        colormap = plt.get_cmap(colormap)

        # No ticks, we'll put our own
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # Set y axes limit, add additonal space if requested
        self.ax.set_ylim(0, 10 + padding)
        # Remove the black axes border which may obscure the labels
        self.ax.spines['polar'].set_visible(False)

        # Draw lines between connected nodes, only draw the strongest connections
        if n_lines is not None and len(con) > n_lines:
            con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
        else:
            con_thresh = 0.

        # get the connections which we are drawing and sort by connection strength
        # this will allow us to draw the strongest connections first
        con_abs = np.abs(con)
        con_draw_idx = np.where(con_abs >= con_thresh)[0]

        con = con[con_draw_idx]
        con_abs = con_abs[con_draw_idx]
        indices = [ind[con_draw_idx] for ind in indices]

        # now sort them
        sort_idx = np.argsort(con_abs)
        con_abs = con_abs[sort_idx]
        con = con[sort_idx]
        indices = [ind[sort_idx] for ind in indices]

        # Get vmin vmax for color scaling
        if vmin is None:
            vmin = np.min(con[np.abs(con) >= con_thresh])
        if vmax is None:
            vmax = np.max(con)
        vrange = vmax - vmin

        self.vrange = [vmin, vmax]
        # We want to add some "noise" to the start and end position of the
        # edges: We modulate the noise with the number of connections of the
        # node and the connection strength, such that the strongest connections
        # are closer to the node center
        nodes_n_con = np.zeros((n_nodes), dtype=np.int)
        for i, j in zip(indices[0], indices[1]):
            nodes_n_con[i] += 1
            nodes_n_con[j] += 1

        # initalize random number generator so plot is reproducible
        rng = np.random.mtrand.RandomState(seed=0)

        n_con = len(indices[0])
        noise_max = 0.25 * node_width
        start_noise = rng.uniform(-noise_max, noise_max, n_con)
        end_noise = rng.uniform(-noise_max, noise_max, n_con)

        nodes_n_con_seen = np.zeros_like(nodes_n_con)
        for i, (start, end) in enumerate(zip(indices[0], indices[1])):
            nodes_n_con_seen[start] += 1
            nodes_n_con_seen[end] += 1

            start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start]) /
                               float(nodes_n_con[start]))
            end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end]) /
                             float(nodes_n_con[end]))

        # scale connectivity for colormap (vmin<=>0, vmax<=>1)
        con_val_scaled = (con - vmin) / vrange

        # Finally, we draw the connections
        for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
            # Start point
            t0, r0 = node_angles[i], 10

            # End point
            t1, r1 = node_angles[j], 10

            # Some noise in start and end point
            t0 += start_noise[pos]
            t1 += end_noise[pos]

            verts = [(t0, r0), (t0, 5), (t1, 5), (t1, r1)]
            codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                     m_path.Path.LINETO]
            path = m_path.Path(verts, codes)

            color = colormap(con_val_scaled[pos])

            # Actual line
            patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                        linewidth=linewidth, alpha=1.)
            self.ax.add_patch(patch)

        # Draw ring with colored nodes
        height = np.ones(n_nodes) * 1.0
        bars = self.ax.bar(node_angles, height, width=node_width, bottom=9,
                           edgecolor=node_edgecolor, lw=node_linewidth,
                           facecolor='.9', align='center')

        for bar, color in zip(bars, node_colors):
            bar.set_facecolor(color)
            #bar.set_edgecolor(color)

        # Draw node labels
        if names:
            angles_deg = 180 * node_angles / np.pi
            for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
                if angle_deg >= 270:
                    ha = 'left'
                else:
                    # Flip the label, so text is always upright
                    angle_deg += 180
                    ha = 'right'

                self.ax.text(angle_rad, 10.4, name, size=fontsize_names,
                             rotation=angle_deg, rotation_mode='anchor',
                             horizontalalignment=ha, verticalalignment='center',
                             color=textcolor)

        self.ax.text(0.0,1.0,'L',transform=self.ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left')
        self.ax.text(1.0,1.0, 'R', transform=self.ax.transAxes,
                     fontsize=8, fontweight='bold', va='top', ha='right')
