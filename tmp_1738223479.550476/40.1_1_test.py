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

    # Handle edge cases by simulating ghost cells with conditional logic
    if target == 0:
        left = u[0]  # Use the value at the boundary for the left ghost cell
    else:
        left = u[target - 1]

    if target == len(u) - 1:
        right = u[-1]  # Use the value at the boundary for the right ghost cell
    else:
        right = u[target + 1]

    # Calculate the second order derivative using the centered difference formula
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