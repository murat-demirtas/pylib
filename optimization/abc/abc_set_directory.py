from tools.io import Data
import sys

"""
Main script
"""
if __name__ == '__main__':
    """
    arguments:
    1- model type: heterogeneous or homogeneous
    2- hemi: L, R or LR
    3- session: session_1, session_2 or all
    4- linear myelin
    5- thickness
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

    if int(arguments[3]) == 1:
        gradient = 'linearized'
    else:
        gradient = 'raw'

    measure = arguments[4] + '_' + arguments[5]

    gradient_direction = arguments[6]

    n_samples = int(arguments[7])

    data = Data()
    data.append_to_output('abc_' + m_type + '_' + hemi + '_' + session + '_' + measure + '_' + gradient + '_' + gradient_direction + '_' + str(n_samples))
