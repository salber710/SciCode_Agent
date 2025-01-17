import numpy as np



# Background: The second order derivative of a function can be approximated using finite difference methods. 
# The central difference scheme is a popular method for this purpose due to its balance between accuracy and computational cost.
# For a second order accurate central difference scheme, the second derivative at a point can be approximated using the values of the function at neighboring points.
# The formula for the second derivative using a central difference scheme is:
# f''(x) ≈ (f(x+dx) - 2*f(x) + f(x-dx)) / (dx^2)
# In this implementation, we use ghost cells to handle boundary conditions. Ghost cells are extra cells added to the computational domain to simplify the handling of boundary conditions.
# Here, the ghost cells are set to have the same value as the nearest boundary cell, which is a simple form of boundary condition.


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
    # Set ghost cells
    u_extended[0] = u[0]  # Left ghost cell
    u_extended[-1] = u[-1]  # Right ghost cell
    
    # Calculate the second derivative using the central difference scheme
    deriv = (u_extended[target + 2] - 2 * u_extended[target + 1] + u_extended[target]) / (dx ** 2)
    
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
