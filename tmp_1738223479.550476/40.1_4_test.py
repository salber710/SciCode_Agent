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

    n = len(u)
    
    # Define a helper function to compute ghost cell values using linear extrapolation
    def ghost_value(index, boundary_index):
        if index < 0:
            return 2 * u[boundary_index] - u[boundary_index + 1]
        elif index >= n:
            return 2 * u[boundary_index] - u[boundary_index - 1]
        else:
            return u[index]
    
    # Compute left and right indices for the centered difference
    left_index = target - 1
    right_index = target + 1

    # Handle boundary conditions using the linear extrapolation for ghost cells
    left_value = ghost_value(left_index, 0)
    right_value = ghost_value(right_index, n - 1)

    # Calculate the second order derivative using the centered difference formula
    deriv = (right_value - 2 * u[target] + left_value) / (dx * dx)

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