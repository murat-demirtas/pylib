import numpy as np
from scipy.optimize import leastsq
import h5py


def make_weight_matrix(d, d0, identity):
        w = np.exp(-d/d0)
        w[identity] = 0 # Mask out diagonal elements (no self-coupling)
        #w /= w.sum(axis=1)  # normalize weight matrix
        return w


def fit_parameters(distance, myelin):
    def func(params):
        rho, d0 = params
        w = make_weight_matrix(distance, d0)
        return myelin - rho * w.dot(myelin)
    x0 = np.array([1., 20]) # initial parameters estimates (rho, d0)
    x, cov_x, infodict, mesg, ier = leastsq(func, x0=x0, full_output=True)

    return x


def generate_surrogate(k, n_roi):
    # Generate random vector of normally distributed samples
    u = np.random.randn(n_roi)

    # Introduce spatial structure into random variates
    x = k.dot(u)

    x -= x.mean()
    x /= x.std()

    return x


def gen_surrogate_maps(distance_matrix, map0, fname, N=20):
    #np.random.seed(137)

    # Compute pairwise geodesic distances between parcel centroids
    # parcel_centroids = human.parcel_centroid_spherical_coords()
    # distance_matrix1 = geodesic_distance(parcel_centroids, parcel_centroids)
    n_roi = len(map0)
    identity = np.eye(n_roi, dtype=bool)

    map0 = (map0 - map0.mean()) / map0.std()
    rho, d0 = fit_parameters(distance_matrix, map0)

    # Construct weight matrix
    w = make_weight_matrix(distance_matrix, d0, identity)

    # Introduce spatial structure into random variates
    k_factor = np.linalg.inv(identity - rho * w)

    print "Myelin map: rho = %f, d0 = %f" % (rho, d0)
    maps = np.empty((N+1, n_roi))
    maps[0] = map0
    for i in range(N): maps[i + 1] = generate_surrogate(k_factor, n_roi)

    surrogate_file = h5py.File(fname, 'w')
    surrogate_file.create_dataset('rho', data=rho)
    surrogate_file.create_dataset('d0', data=d0)
    surrogate_file.create_dataset('distance_matrix', data=distance_matrix)
    surrogate_file.create_dataset('surrogates', data=maps)
    surrogate_file.close()

    return

def moran(map, w):
    N = map.shape[0]

    map_mat = np.tile(map, (N, 1))
    x = np.abs(map_mat - map_mat.T)

    xu = N*(w*(x-x.mean()).dot(x.T - x.mean())).sum()
    xl = w.sum()*((x - x.mean())**2).sum()
    return xu/xl


def fit_moran(distance, myelin):
    def func(rho):
        return myelin - rho * distance.dot(myelin)

    x0 = np.array([0.5]) # initial parameters estimates (rho, d0)
    x, cov_x, infodict, mesg, ier = leastsq(func, x0=x0, full_output=True)
    return x



