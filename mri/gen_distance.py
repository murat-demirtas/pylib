import numpy as np
from utils.cifti import Gifti
from os import path, system, pardir

def average_parcelwise_euclidean_distance(surface_file, parcel_file, output_fname):
    module_path = path.dirname(path.realpath(__file__))
    temp_path = module_path + '/tmp/'
    parent_path = path.abspath(path.join(module_path, pardir))

    surface_file = parent_path + '/data/templates/templates_32k/surface/' + surface_file + '.surf.gii'

    coordinate_metric_file = temp_path + "vertex_coordinates.func.gii"
    system(  # Create metric file containing coordinates of each surf vertex
        'wb_command -surface-coordinates-to-metric "%s" "%s"' % (
            surface_file, coordinate_metric_file))

    coordinates_file = Gifti(coordinate_metric_file)
    coordinates = np.vstack((coordinates_file.data(0), coordinates_file.data(1), coordinates_file.data(2)))

    parcel_info = Gifti(parcel_file)
    labels = parcel_info.data(0)
    unique_labels = np.unique(labels)[1:]  # skip label 0
    n_rois = len(unique_labels)
    distance_matrix = np.zeros((n_rois, n_rois))

    parcel_vertex_mask = {l:labels==l for l in unique_labels}
    for i, li in enumerate(unique_labels[:-1]):
        other_labels = unique_labels[i+1:]
        coord_i = coordinates[:,parcel_vertex_mask[li]]
        N_i = coord_i.shape[1]
        # Append distances from vertex i to every vertex in label j
        for j, lj in enumerate(other_labels):
            coord_j = coordinates[:, parcel_vertex_mask[lj]]
            N_j = coord_j.shape[1]
            distance_matrix[i, i + j + 1] = np.sqrt(((np.tile(coord_i, (N_j, 1, 1)) - np.tile(coord_j, (N_i, 1, 1)).T) ** 2).sum(1)).mean()
        print "## Parcel label %s complete." % str(li)

    # Make symmetric
    i,j = np.triu_indices(n_rois, k=1)
    distance_matrix[j,i] = distance_matrix[i,j]

    np.save(output_fname, distance_matrix)

    remove(coordinate_metric_file)
    return



def average_parcelwise_geodesic_distance(surface_file, parcel_file, output_fname):
    module_path = path.dirname(path.realpath(__file__))
    temp_path = module_path + '/tmp/'
    parent_path = path.abspath(path.join(module_path, pardir))

    surface_file = parent_path + '/data/templates/templates_32k/surface/' + surface_file + '.surf.gii'

    coordinate_metric_file = temp_path + "vertex_coordinates.func.gii"
    system(  # Create metric file containing coordinates of each surf vertex
        'wb_command -surface-coordinates-to-metric "%s" "%s"' % (
            surface_file, coordinate_metric_file))

    distance_metric_file = temp_path + "geodesic_distance.func.gii"

    parcel_info = Gifti(parcel_file)
    labels = parcel_info.data(0)
    unique_labels = np.unique(labels)[1:]  # skip label 0
    n_rois = len(unique_labels)
    distance_matrix = np.zeros((n_rois, n_rois))

    parcel_vertex_mask = {l:labels==l for l in unique_labels}
    for i, li in enumerate(unique_labels[:-1]):
        other_labels = unique_labels[i+1:]

        # Initialize empty lists
        parcel_distances = {lj: [] for lj in other_labels}

        # For every vertex in this parcel
        li_vertices = np.where(parcel_vertex_mask[li])[0]
        for vi in li_vertices:

            # Compute the geodesic distance to every other vertex
            system(
                'wb_command -surface-geodesic-distance "%s" %i "%s" ' % (
                surface_file, vi, distance_metric_file))

            # Load geodesic distances from vertex i
            dist_metric = Gifti(distance_metric_file)
            all_distances = dist_metric.data(0)

            # Append distances from vertex i to every vertex in label j
            for lj in other_labels:
                vi_lj_distances = all_distances[parcel_vertex_mask[lj]]
                parcel_distances[lj].append(vi_lj_distances)

        # Average distances and record in matrix
        for j, lj in enumerate(other_labels):
            mean_distance = np.mean(parcel_distances[lj])
            distance_matrix[i,i+j+1] = mean_distance

        print "## Parcel label %s complete." % str(li)
        print "## Distances: ", distance_matrix[i]

    remove(coordinate_metric_file)
    remove(distance_metric_file)

    # Make symmetric
    i,j = np.triu_indices(n_rois, k=1)
    distance_matrix[j,i] = distance_matrix[i,j]

    np.save(output_fname, distance_matrix)
    return


def remove(file):
    system('rm ' + file)