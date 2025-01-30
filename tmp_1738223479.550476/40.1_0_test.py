from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The second-order derivative of a function can be approximated using a centered finite difference scheme. 
# This approach is often used in numerical methods to approximate derivatives when the exact form of the function is unknown 
# or difficult to differentiate analytically. The centered finite difference method uses values at surrounding points to 
# approximate the derivative at a point of interest. The second-order centered difference formula is given by:
# f''(x) ≈ (f(x+dx) - 2*f(x) + f(x-dx)) / (dx^2),
# where dx is the spacing between points. To handle boundary conditions, ghost cells can be utilized, which extend the 
# grid beyond its physical boundaries. In this implementation, ghost cells are used with values equal to the nearest 
# cell on the boundary, which helps in maintaining the scheme's symmetry and accuracy at the boundaries.

def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''

    
    # Check if the target index is within the bounds of the array
    if target < 0 or target >= len(u):
        raise IndexError("Target index is out of bounds.")
    
    # Create a copy of the array with ghost cells added at both ends
    extended_u = np.zeros(len(u) + 2)
    
    # Assign values to the ghost cells (equal to the nearest cell on the boundary)
    extended_u[0] = u[0]
    extended_u[-1] = u[-1]
    
    # Copy the original array into the extended array
    extended_u[1:-1] = u
    
    # Calculate the second derivative using the centered difference formula
    deriv = (extended_u[target + 2] - 2 * extended_u[target + 1] + extended_u[target]) / (dx * dx)
    
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