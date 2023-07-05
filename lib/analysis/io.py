#! usr/bin/python
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd
import h5py


class Data:
    def __init__(self, input_path=None, output_path=None, mkdir=False):

        self._input_path = Path() if input_path is None else Path(input_path)
        self._output_path = Path() if output_path is None else Path(output_path)

        if not self._input_path.exists():
            if mkdir:
                print("Creating input path " + str(self._input_path))
                self._input_path.mkdir()
            else:
                raise Exception("Input path does not exist.")

        if not self._output_path.exists():
            if mkdir:
                print("Creating output path " + str(self._output_path))
                self._output_path.mkdir()
            else:
                raise Exception("Output path does not exist.")

    def change_path(self, input_path=True, directory=None, mkdir=True):
        if input_path:
            if directory is None:
                self._input_path = self._input_path.parent
            else:
                self._input_path = self._input_path.joinpath(directory)
                if not self._input_path.exists():
                    if mkdir:
                        print("Creating input path " + str(self._input_path))
                        self._input_path.mkdir()
                    else:
                        raise Exception("Input path does not exist.")
        else:
            if directory is None:
                self._output_path = self._output_path.parent
            else:
                self._output_path = self._output_path.joinpath(directory)
                if not self._output_path.exists():
                    if mkdir:
                        print("Creating output path " + str(self._output_path))
                        self._output_path.mkdir()
                    else:
                        raise Exception("Output path does not exist.")

    def load(self, filename, input_path=True, numeric=False, *args, **kwargs):
        if input_path:
            file_path = self._input_path.joinpath(filename)
        else:
            file_path = self._output_path.joinpath(filename)

        file_extension = file_path.suffix

        if file_extension == '.hdf5':
            return h5py.File(file_path, 'r')
        elif file_extension == '.mat':
            return loadmat(file_path)
        elif (file_extension == '.npy') | (file_extension == '.npz'):
            return np.load(file_path)
        elif file_extension == '.xlsx':
            return pd.read_excel(file_path, *args, **kwargs)
        elif file_extension == '.csv':
            if numeric:
                return np.loadtxt(file_path)
            else:
                return pd.read_csv(file_path, *args, **kwargs)
        elif file_extension == '.txt':
            if numeric:
                return np.loadtxt(file_path, *args, **kwargs)
            else:
                f = open(file_path, 'r')
                read_data = f.readlines()
                return [read_data[ii][:-1] for ii in range(len(read_data))]
        else:
            raise ValueError('File extension should be hdf5, mat, npy, xlsx, csv or txt')

    def save(self, filename, data=None, numeric=False, *args, **kwargs):
        file_path = self._output_path.joinpath(filename)
        file_extension = file_path.suffix

        if file_extension == '.hdf5':
            return h5py.File(file_path, 'w')
        elif file_extension == '.mat':
            savemat(file_path, data)
        elif (file_extension == '.npy') | (file_extension == '.npz'):
            np.save(file_path, data)
        elif file_extension == '.xlsx':
            data.to_excel(file_path, *args, **kwargs)
        elif file_extension == '.csv':
            if numeric:
                np.savetxt(file_path, data)
            else:
                data.to_csv(file_path, *args, **kwargs)
        elif file_extension == '.txt':
            np.savetxt(file_path, data)
        else:
            raise ValueError('File extension should be hdf5, mat, npy, xlsx, csv or txt')

    @property
    def input_path(self):
        return str(self._input_path)

    @property
    def output_path(self):
        return str(self._output_path)
