import pandas as pd
import numpy as np
import h5py

def init_vars(keys, size, my_dict=None):
    if my_dict is None:
        my_dict = {}
    for ii in range(len(keys)):
        my_dict[keys[ii]] = np.empty(size)
    return my_dict


def tabulate_results(values, rows, columns):
    dummy_values = pd.DataFrame(values, columns=columns)
    return pd.concat([rows, dummy_values], axis=1)


def to_dict(values, keys, pandize=False):
    my_dict = {}
    for ii in range(len(values)):
        my_dict[keys[ii]] = values[ii]
    if pandize:
        my_dict = pd.DataFrame.from_dict(my_dict)
    return my_dict


def save_hdf(data, groups, keys, fname, io_class = None):
    if io_class is None:
        fout = h5py.File(fname, 'w')
    else:
        fout = io_class.save(fname)

    gr = []
    for gi, gg in enumerate(groups):
        gr += [fout.create_group(gg)]
        for ki, kk in enumerate(keys):
            gr[-1].create_dataset(kk, data=data[gi][ki], compression="lzf")
    fout.close()


def load_hdf(fname, group=None, keys=None, todict=True, io_class=None, from_path=None):
    if io_class is None:
        fin = h5py.File(fname, 'r')
    else:
        fin = io_class.load(fname, from_path=from_path)

    if todict:
        if keys is None:
            keys = fin.keys()
        out = {}
        for kk in keys:
            if group is not None:
                out[kk] = fin[group][kk].value
            else:
                out[kk] = fin[kk].value
        fin.close()
    else:
        if keys is None:
            keys = fin.keys()
        out = []
        for kk in keys:
            if group is not None:
                out += [fin[group][kk].value]
            else:
                out += [fin[group][kk].value]
        fin.close()

    return out