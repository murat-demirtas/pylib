import numpy as np
from .hopf import HopfModel


class Hopf():
    """
    Wrapper class for Hopf model.
    """

    def __init__(self, sc, f_diff=None, gradient=None, *args, **kwargs):
        """

        Parameters
        ----------
        sc : ndarray
            Structural connectivity matrix
        gradient : ndarray, optional
            Heterogeneity map to scale local model parameters. 
            If None, the model parameters are homogeneous (None by default)

        Notes
        -----
        The optional arguments and keyword arguments pass to the Model class.
        If the model is separated for left and right hemispheres, the SC and heterogeneity map
         should be given as a list as [left, right]. In this case, the wrapper generates 2 models
         as a list in the same order that is [left, right].

        """
        self.sc = sc
        self.gradient = gradient
        if not f_diff:
            f_diff = np.ones(self.sc.shape[0]) * 0.05

        if isinstance(self.sc, list):
            if not isinstance(gradient, list):
                self.gradient = [self.gradient, self.gradient]
            self.hopf = [HopfModel(self.sc[ii], f_diff = f_diff, hmap=self.gradient[ii], g=1.0,
                              *args, **kwargs) for ii in range(2)]
        else:
            self.hopf = HopfModel(self.sc, f_diff = f_diff, g=1.0, hmap=self.gradient,
                             *args, **kwargs)

    def set(self, parameter, values, separate=False):
        """
        Set method for the model

        Parameters
        ----------
        parameter : str
            The name of the parameter, such as 'a' or 'g'
        values : float or tuple
            The parameter values to set. The values should be tuple for heterogeneous model.
        separate : bool, optional
            If True, the value passes to left and right hemisphere separately. This flag is 
             required only if the global coupling parameter is separate for each hemisphere. 
        """

        if isinstance(self.hopf, list):
            if separate:
                [setattr(self.hopf[ii], parameter, values[ii]) for ii in range(2)]
            else:
                [setattr(self.hopf[ii], parameter, values) for ii in range(2)]
        else:
            setattr(self.hopf, parameter, values)

    def get(self, parameter):
        """
        Get method for the model

        Parameters
        ----------
        parameter : str
            The parameter to get, such as 'w_EE', 'w_EI', 'corr_bold'...etc. 
        """

        if isinstance(self.hopf, list):
            return [self.hopf[ii].__getattribute__(parameter) for ii in range(2)]
        else:
            return self.hopf.__getattribute__(parameter)

    def check_stability(self):
        """
        Check stability of the system

        Parameters
        ----------
        compute_FIC : bool, optional
            If True, the feedback inhibition is recomputed.

        Returns
        -------
        bool
            The truth value for the largest eigenvalue of the system is smaller than 0

        """

        if isinstance(self.hopf, list):
            [self.hopf[ii]._compute_jacobian() for ii in range(2)]
            return np.array([self.hopf[ii]._unstable for ii in range(2)]).any()
        else:
            self.hopf._compute_jacobian()
            return self.hopf._unstable

    def moments_method(self, *args, **kwargs):
        """
        Calls moments_method method in the Model class

        """

        if isinstance(self.hopf, list):
            [self.hopf[ii].moments_method(*args, **kwargs) for ii in range(2)]
        else:
            self.hopf.moments_method(*args, **kwargs)
