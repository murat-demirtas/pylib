from tools.io import Data
from tools.linalg import subdiag
import numpy as np
import sys
from scipy import stats
from aux.toolbox import Toolbox
from aux.bayesian_sampler import Model

"""
Main script
"""
if __name__ == '__main__':
    """
    arguments:
    1- model type: heterogeneous or homogeneous
    2- hemi: L, R, LR or whole
    3- session: session_1, session_2 or all
    4- linearize (1), raw (0)
    5- measure thickness or myelin
    6- minimum number of samples
    7- number of iterations
    8- iteration count
    """

    arguments = sys.argv[1]
    arguments = arguments.split("_")

    m_type = arguments[0]
    hemi = arguments[1]
    if int(arguments[2]) == 1:
        session = 'session_1'
    elif int(arguments[2]) == 2:
        session = 'session_2'
    else:
        session = 'all'

    linearized = False
    if int(arguments[3]) == 1:
        gradient = 'linearized'
        linearized = True
    else:
        gradient = 'raw'

    measure = arguments[4] + '_' + arguments[5]

    gradient_direction = arguments[6]

    n_samples = int(arguments[7])
    iteration = int(arguments[8])
    run_id = int(arguments[9])

    myelin = None
    thickness = None

    """
    Prepare Data
    """
    emp_data = Toolbox()
    emp_data.load(measure, linearized=linearized, hemi=hemi, session=session)

    fc = emp_data.fc
    sc = emp_data.sc
    #myelin = np.sqrt(emp_data.myelin)
    myelin = emp_data.myelin
    thickness = emp_data.thickness

    if isinstance(sc, list):
        rejection_threshold = 1.0 - np.corrcoef(np.hstack((subdiag(sc[0]), subdiag(sc[1]))), fc.mean(1))[0, 1]
    else:
        rejection_threshold = 1.0 - np.corrcoef(subdiag(sc), fc.mean(1))[0, 1]

    """
    set output data
    """
    data = Data()
    data.append_to_output('abc_' + m_type + '_' + hemi + '_' + session + '_' + measure + '_' + gradient + '_' + gradient_direction + '_' + str(n_samples))



    """
    Single iteration of Approximate Bayesian Optimization 
    """
    if m_type == 'heterogeneous':
        model = Model(sc, fc, myelin=myelin, thickness=thickness, heterogeneity=True, gradient_dir=gradient_direction, min_samples=n_samples)
        if hemi == 'LR':
            prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5), stats.uniform(0.001, 5.0), stats.uniform(0.0, 15.0), stats.uniform(0.001, 5.0),
                     stats.uniform(0.001, 5.0)]
        else:
            prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5), stats.uniform(0.001, 5.0), stats.uniform(0.0, 15.0),
                     stats.uniform(0.001, 5.0)]
        model.set_prior(prior)
    else:
        model = Model(sc, fc, myelin=None, thickness=None, heterogeneity=False, min_samples=n_samples)
        if hemi == 'LR':
            prior = [stats.uniform(0.001, 5.), stats.uniform(0.001, 15.0), stats.uniform(0.001, 5.0), stats.uniform(0.001, 5.0)]
        else:
            prior = [stats.uniform(0.001, 5.), stats.uniform(0.001, 15.0), stats.uniform(0.001, 5.0)]
        model.set_prior(prior)


    """
    Write output
    """
    if iteration > 1:
        file_prev = data.load('iteration_' + str(iteration-1) + '.hdf5', from_path=data.output_dir)
        theta_prev = file_prev['theta'].value
        weights_prev = file_prev['weights'].value
        tau_squared_prev = file_prev['tau_squared'].value
        epsilon = file_prev['epsilon'].value
        file_prev.close()
        results = model.run_sampler(epsilon, theta_prev=theta_prev, weights=weights_prev, tau_squared=tau_squared_prev, first_sample=False)
    else:
        results = model.run_sampler(rejection_threshold, first_sample=True)

    data.save('samples_'+str(run_id)+'.npy', results)

