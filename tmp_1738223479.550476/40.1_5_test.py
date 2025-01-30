from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''
    
    # Get the length of the array
    n = len(u)

    # Check for boundary conditions and use symmetric extension for ghost cells
    if target <= 0:
        left = u[0]  # Symmetric extension for left boundary
    else:
        left = u[target - 1]

    if target >= n - 1:
        right = u[n - 1]  # Symmetric extension for right boundary
    else:
        right = u[target + 1]

    # Use the centered difference formula for second order derivative
    deriv = (right - 2 * u[target] + left) / (dx * dx)

    return deriv


try:
    targets = process_hdf5_to_tuple('40.1', 3)
    target = targets[0]
    target_ = 0
    u = np.array([-1,-1, -1, 0,1,2,3,4,5,6])
    dx = 0.1
    assert np.allclose(second_diff(target_, u, dx), target)

    target = targets[1]
    target_ = 2
    u = np.array([0,1,2,3,4,5,6])
    dx = 0.1
    assert np.allclose(second_diff(target_, u, dx), target)

    target = targets[2]
    u = np.array([0,1,2,4,6,8,0,1,23])
    target_ = u.size-1
    dx = 0.1
    assert np.allclose(second_diff(target_, u, dx), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e