import numpy as np



# Background: 
# The second derivative of a function gives information about its concavity and inflection points. 
# In numerical analysis, the second derivative can be approximated using finite difference methods.
# The central difference scheme is a common method to approximate derivatives, offering a good balance 
# between accuracy and computational cost. For second-order accuracy, the central difference formula for 
# the second derivative at a point i is given by:
# f''(x_i) ≈ (f(x_{i+1}) - 2*f(x_{i}) + f(x_{i-1})) / (dx^2)
# where dx is the spacing between points.
# For boundary conditions, ghost cells are used, where the value of a ghost cell is set equal to its 
# nearest boundary cell to maintain the same conditions at the edges, which helps in simplifying 
# calculations without additional boundary-specific logic.

def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''

    
    # Ensure the array length is adequate for the central difference calculation
    if len(u) < 5:
        raise ValueError("Array u must have a minimum length of 5.")
    
    # Calculate the second derivative using central difference
    if target == 0:  # Left boundary (using ghost cell)
        # Use u[0] itself as the ghost cell value
        deriv = (u[1] - 2 * u[0] + u[0]) / (dx**2)
    elif target == len(u) - 1:  # Right boundary (using ghost cell)
        # Use u[-1] itself as the ghost cell value
        deriv = (u[-1] - 2 * u[-1] + u[-2]) / (dx**2)
    else:
        # Central difference scheme for internal points
        deriv = (u[target + 1] - 2 * u[target] + u[target - 1]) / (dx**2)
    
    return deriv

from scicode.parse.parse import process_hdf5_to_tuple
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
