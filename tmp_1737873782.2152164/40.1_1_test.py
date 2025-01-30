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
    
    # Ensure that the array has at least 5 elements
    assert len(u) >= 5, "Array u must have at least 5 elements"
    
    # Handle ghost cells by mirroring the edge values
    if target == 0:
        # Use ghost cell at the left end
        T_im1 = u[0]
        T_i = u[0]
        T_ip1 = u[1]
    elif target == len(u) - 1:
        # Use ghost cell at the right end
        T_im1 = u[-2]
        T_i = u[-1]
        T_ip1 = u[-1]
    else:
        # Use normal central difference scheme for interior points
        T_im1 = u[target - 1]
        T_i = u[target]
        T_ip1 = u[target + 1]
    
    # Calculate the second order derivative using the central difference formula
    deriv = (T_ip1 - 2 * T_i + T_im1) / (dx ** 2)
    
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