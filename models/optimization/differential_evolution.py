import numpy as np
from scipy.optimize import differential_evolution
from abc import ABCMeta, abstractmethod


class Dev:
    __metaclass__ = ABCMeta
    def __init__(self, parameters, bounds, verbose=True):

        self.verbose = verbose
        self.set_model()

        self.parameters = parameters
        self.bounds = []
        for ii in range(len(parameters)):
            self.bounds += bounds[parameters[ii]]

    @abstractmethod
    def set_model(self):
        """

        :return:
        """

    @abstractmethod
    def sample(self, theta):
        """

        :param theta:
        :return:
        """

    def opt_func(self, theta):
        self.sample(theta)
        return self.cost()

    def run(self, filename):
        results = differential_evolution(self.opt_func, self.mybounds, disp=True)
        opt_params = results.x
        opt_fit = -results.fun

        data = Data()
        fout = data.save(filename)
        fout.create_dataset('optimal_fit', data=opt_fit)
        fout.create_dataset('optimal_parameters', data=opt_params)
        fout.close()
