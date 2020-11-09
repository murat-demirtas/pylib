import numpy as np
import h5py
from scipy.optimize import differential_evolution
from abc import ABCMeta, abstractmethod


class Dev:
    __metaclass__ = ABCMeta
    def __init__(self, model, objective, pdict, verbose=True):

        self.verbose = verbose

        self.model = model
        self.objective = objective
        self.parameter_dictionary = pdict
        self.parameters = list(self.parameter_dictionary.keys())
        self.bounds = []
        for i in range(len(self.parameters)):
            self.bounds += self.parameter_dictionary[self.parameters[i]]
            
        self.opt_params = None
        self.opt_fit = None

    @abstractmethod
    def cost_function(self):
        """

        :return: cost to minimize
        """
        pass
        
    def sample(self, theta):
        """
        Updates model given the sample
        
        :param theta:
        :return:
        """
        pind = 0
        for p in self.parameters:
            pl = len(self.parameter_dictionary[p])
            setattr(self.model, p, theta[pind:pind+pl])
            pind+=pl

    def opt_func(self, theta):
        self.sample(theta)
        return self.cost_function()

    def run(self, filename = None, **kwargs):
        self.results = differential_evolution(self.opt_func, self.bounds, disp=self.verbose, **kwargs)
        self.opt_params = self.results.x
        self.opt_fit = -self.results.fun

        if filename is not None:
            fout = h5py.File(filename, 'w')
            fout.create_dataset('optimal_fit', data=self.opt_fit)
            fout.create_dataset('optimal_parameters', data=self.opt_params)
            fout.close()
