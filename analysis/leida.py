from scipy.spatial.distance import squareform
import sklearn.cluster as sl
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
from scipy import stats, signal, special
from analysis.cifti import Cifti
from analysis import utils
from analysis import signal as asp
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples
import matplotlib.cm as cm
from scipy.spatial.distance import cosine

def get_leading_eigenvectors(plvs):
    """
    Get Leading eigenvectors for each timepoint given
        the phase coherence matrices
    """
    dummy, ref_evect = np.linalg.eig(squareform(plvs.mean(1)))

    pmodes = np.empty((360, plvs.shape[1]))
    for i in range(plvs.shape[1]):
        evals, evects = np.linalg.eig(squareform(plvs[:,i]))
        if cosine(ref_evect[:,0], evects[:,0]) < cosine(ref_evect[:,0], -evects[:,0]):
            pmodes[:,i] = evects[:,0].real
        else:
            pmodes[:, i] = -evects[:, 0].real
    return pmodes

def clustering(data, m):
    """
    Perform K-means clustering on the data
    """
    kmeans = sl.KMeans(m)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    cluster_centers = kmeans.cluster_centers_
    return clusters, cluster_centers

def plot_nclusters(leading_eigs, cmax=12):
    db_score = np.empty(cmax - 2)
    silhouette = np.empty(cmax - 2)
    for i, n in enumerate(range(2, cmax)):
        kmeans = sl.KMeans(n)
        kmeans.fit(leading_eigs.T)
        clusters = kmeans.labels_
        db_score[i] = davies_bouldin_score(leading_eigs.T, clusters)
        silhouette[i] = silhouette_score(leading_eigs.T, clusters)

    f, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].plot(range(2, cmax), db_score, lw=1.5)
    ax[0].set_xlabel('number of clusters')
    ax[0].set_title('Davies Bouldin Score')
    ax[1].plot(range(2, cmax), silhouette, lw=1.5)
    ax[1].set_xlabel('number of clusters')
    ax[1].set_title('Silhouette Score')


def silhouette_values(leading_eigs, clusters, cluster_centers, n_clusters):
    cc = cluster_centers
    sample_silhouette_values = silhouette_samples(leading_eigs.T, clusters)
    cluster_labels = clusters

    f, ax1 = plt.subplots(1, 1, figsize=(4, 3))
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

def pms(clusters, n_clusters, t=4800):
    prob_cluster = np.array([(clusters == i).mean() for i in range(n_clusters)])

    transitions = np.empty((n_clusters, n_clusters))
    for i in range(n_clusters):
        indices = np.where(clusters == i)[0] + 1
        if (indices == t).any():
            indices = indices[indices != t]
        transitions[i, :] = np.array([(clusters[indices] == j).mean() for j in range(n_clusters)])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.pcolormesh(transitions)

    ax1.set_yticks(np.arange(n_clusters) + 0.5)
    ax1.set_xticks(np.arange(n_clusters) + 0.5)

    ax1.set_yticklabels(np.arange(1, n_clusters+1))
    ax1.set_xticklabels(np.arange(1, n_clusters+1))

    ax1.set_title('Transition Probability Matrix')

    for i in range(n_clusters):
        for j in range(n_clusters):
            ax1.text(j + 0.5, i + 0.5, str('{:3.2f}'.format(transitions[i, j])), va='center', ha='center',
                     fontweight='bold', color='w')

    ax2.bar(range(1, n_clusters+1), prob_cluster)
    ax2.set_xticks(np.arange(1, n_clusters+1))
    ax2.set_xticklabels(np.arange(1, n_clusters+1))
    ax2.set_title('Probability of Clusters')

    plt.tight_layout()