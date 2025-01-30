from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The second order derivative of a function can be approximated using finite difference methods. 
# The central difference scheme is a popular method for this purpose due to its balance of accuracy and simplicity.
# For a second order accurate central difference scheme, the second derivative at a point can be approximated as:
# f''(x) ≈ (f(x+dx) - 2*f(x) + f(x-dx)) / (dx^2)
# This formula assumes that the function values at x+dx and x-dx are available. In computational grids, 
# especially near boundaries, ghost cells are often used to handle boundary conditions. Here, ghost cells 
# are filled with values equal to the nearest boundary cell to maintain the grid size and apply the central 
# difference scheme uniformly across the domain.

def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''

    
    # Ensure the array is large enough to apply the central difference scheme
    if len(u) < 5:
        raise ValueError("Array u must have at least 5 elements.")
    
    # Create a copy of u with ghost cells
    u_extended = np.zeros(len(u) + 2)
    u_extended[1:-1] = u
    # Fill ghost cells with nearest boundary values
    u_extended[0] = u[0]
    u_extended[-1] = u[-1]
    
    # Calculate the second order derivative using the central difference scheme
    deriv = (u_extended[target + 1] - 2 * u_extended[target] + u_extended[target - 1]) / (dx ** 2)
    
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