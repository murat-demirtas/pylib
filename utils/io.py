#! usr/bin/python
import numpy as np
from scipy.io import loadmat, savemat
import os
from os import system
import pandas as pd
import nibabel as nib
import h5py

class Data:
    def __init__(self, flat = False, input_dir = None, output_dir = None, plot_dir = None):
        self.set_paths()

        if flat:
            output_dir = ''
            plot_dir = ''

        if self.parent_path_name == 'grace':
            plot_dir = ''

        if input_dir == None:
            # Set Default Input Directory
            self.input_dir = self.master_path + '/data/'
        else:
            self.input_dir = input_dir

        if output_dir == None:
            # Set Default Output Directory
            if self.parent_path_name == 'grace':
                self.output_dir = self.master_path + '/output/grace/' + self.project_path_name
            else:
                self.output_dir = self.master_path + '/output/local/' + self.project_path_name
            if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
            self.output_dir += '/'
        else:
            self.output_dir = output_dir

        if plot_dir == None:
            # Set Default Plot Directory
            self.docs_dir = self.master_path + '/docs/' + self.project_path_name
            self.plot_dir = self.master_path + '/docs/' + self.project_path_name + '/fig'
            if not os.path.exists(self.docs_dir): os.makedirs(self.docs_dir)
            if not os.path.exists(self.plot_dir): os.makedirs(self.plot_dir)
            self.plot_dir += '/'
        else:
            self.plot_dir = plot_dir


    ################################################################################################################
    #### DIRECTORY MANAGEMENT
    ################################################################################################################
    def set_paths(self):
        self.current_path = os.getcwd()
        self.project_path_name = os.path.basename(self.current_path)

        self.parent_path = os.path.abspath(os.path.join(self.current_path, os.pardir))
        self.master_path = os.path.abspath(os.path.join(self.parent_path, os.pardir))
        self.parent_path_name = os.path.basename(self.parent_path)

        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.module_parent_path = os.path.abspath(os.path.join(self.module_path, os.pardir))

    def append_to_input(self, directory):
        self.input_dir += directory

    def append_to_output(self, directory):
        if not os.path.exists(self.output_dir + directory): os.makedirs(self.output_dir + directory)
        self.output_dir += directory + '/'

    def append_to_plot(self, directory):
        self.plot_dir += directory

    ################################################################################################################
    #### DATA MANAGEMENT
    ################################################################################################################
    def load(self, filename, numeric=False, type=None, from_path=False, from_output=False, *args, **kwargs):
        if from_path:
            file_path = from_path + filename
        else:
            if from_output:
                file_path = self.output_dir + filename
            else:
                file_path = self.input_dir + filename

        if type is None:
            filename, file_extension = os.path.splitext(file_path)
        else:
            file_extension = type

        if file_extension == '.hdf5':
            return h5py.File(file_path,'r')

        if file_extension == '.mat':
            return loadmat(file_path)

        if (file_extension == '.npy') | (file_extension == '.npz'):
            return np.load(file_path)

        if file_extension == '.xlsx':
            return pd.read_excel(file_path, *args, **kwargs)

        if file_extension == '.csv':
            if numeric:
                return np.loadtxt(file_path)
            else:
                return pd.read_csv(file_path, *args, **kwargs)

        if file_extension == '.txt':
            if numeric:
                return np.loadtxt(file_path, *args, **kwargs)
            else:
                f = open(file_path, 'r')
                read_data = f.readlines()
                return [read_data[ii][:-1] for ii in range(len(read_data))]


    def save(self, filename, data=None, numeric=False, compress=False, *args, **kwargs):
        file_path = self.output_dir + filename
        filename, file_extension = os.path.splitext(file_path)
        if file_extension == '.hdf5':
            return h5py.File(file_path, 'w')

        if file_extension == '.mat':
            savemat(file_path, data)

        if (file_extension == '.npy') | (file_extension == '.npz'):
            np.save(file_path, data)

        if file_extension == '.xlsx':
            data.to_excel(file_path, *args, **kwargs)

        if file_extension == '.csv':
            if numeric:
                np.savetxt(file_path, data)
            else:
                data.to_csv(file_path, *args, **kwargs)

        if file_extension == '.txt':
            np.savetxt(file_path, data)

        if compress: self.gzip(file_path)


    def gzip(self, filename):
        system('gzip ' + filename)


    def uzip(self, filename, outputname=None):
        if outputname is None:
            system('gzip -d ' + filename)
        else:
            system('gzip -cd ' + filename + ' > ' + outputname)


    def to_dict(self, values, keys, use_pandas=False):
        my_dict = {}
        for ii in range(len(values)):
            my_dict[keys[ii]] = values[ii]
        if use_pandas:
            my_dict = pd.DataFrame.from_dict(my_dict)
        return my_dict

    ################################################################################################################
    #### NIFTI MANAGEMENT
    ################################################################################################################
    def load_nifti(self, filename, from_input=False):
        file_path = self.input_dir + filename
        return np.array(np.squeeze(nib.load(file_path).get_data()))

    def save_nifti(self, output, filename, template):
        of = nib.load(template)
        temp_data = np.array(of.get_data())
        data_to_write = output.reshape(np.shape(temp_data))
        new_img = nib.Nifti1Image(data_to_write, affine=of.get_affine(), header=of.get_header())
        nib.save(new_img, self.output_dir + filename)

    ################################################################################################################
    #### CIFTI MANAGEMENT
    ################################################################################################################
    def load_cifti(self, filename, from_input=False):
        if from_input: filename = self.input_dir + filename
        of = nib.load(filename)
        return np.array(np.squeeze(of.get_data()))

    def save_cifti(self, output, filename, template):
        of = nib.load(template)
        data_to_write = output.reshape(np.shape(np.array(of.get_data())))
        new_img = nib.Nifti2Image(data_to_write, affine=of.get_affine(), header=of.get_header())
        nib.save(new_img, filename)