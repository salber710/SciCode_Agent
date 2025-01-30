import numpy as np

# Background: The second order derivative of a function can be approximated using finite difference methods. 
# The central difference scheme is a popular method for this purpose due to its balance of accuracy and simplicity.
# For a second order accurate central difference scheme, the second derivative at a point can be approximated as:
# f''(x) ≈ (f(x+dx) - 2*f(x) + f(x-dx)) / (dx^2)
# This formula uses the values of the function at the point of interest and its immediate neighbors.
# In computational domains with boundaries, ghost cells are often used to handle boundary conditions. 
# Here, ghost cells are assigned the value of the nearest boundary cell to maintain the domain size and apply the scheme uniformly.


def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''
    # Ensure the array has at least 5 elements
    if len(u) < 5:
        raise ValueError("Array u must have at least 5 elements.")
    
    # Ensure dx is positive and non-zero
    if dx <= 0:
        raise ValueError("dx must be a positive, non-zero value.")
    
    # Ensure target is an integer
    if not isinstance(target, int):
        raise TypeError("Target index must be an integer.")
    
    # Ensure all elements in u are numeric
    if not np.issubdtype(u.dtype, np.number):
        raise TypeError("All elements in array u must be numeric.")
    
    # Create a copy of u with ghost cells
    u_extended = np.zeros(len(u) + 2, dtype=u.dtype)
    u_extended[1:-1] = u
    u_extended[0] = u[0]  # Left ghost cell
    u_extended[-1] = u[-1]  # Right ghost cell
    
    # Check if the target index is valid
    if target < 0 or target >= len(u):
        raise IndexError("Target index is out of the valid range.")
    
    # Calculate the second derivative using the central difference scheme
    deriv = (u_extended[target + 1 + 1] - 2 * u_extended[target + 1] + u_extended[target + 1 - 1]) / (dx ** 2)
    
    return deriv



# Background: Strang splitting is a numerical method used to solve partial differential equations (PDEs) by splitting the problem into simpler sub-problems that can be solved sequentially. 
# It is particularly useful for problems involving both advection and diffusion processes. The method involves splitting the time evolution operator into two or more parts, 
# each of which can be solved more easily. For a PDE involving diffusion, the Strang splitting method can be applied by first solving the diffusion part for half a time step, 
# then solving the advection part for a full time step, and finally solving the diffusion part again for half a time step. This approach helps in maintaining the accuracy and stability 
# of the numerical solution over time.


def Strang_splitting(u, dt, dx, alpha):
    '''Inputs:
    u : solution, array of float
    dt: time interval , float
    dx: spatial interval, float
    alpha: diffusive coefficient, float
    Outputs:
    u : solution, array of float
    '''
    # Number of spatial points
    n = len(u)
    
    # Create a copy of u to store the updated solution
    u_new = np.copy(u)
    
    # First half-step for diffusion
    for i in range(1, n-1):
        u_new[i] = u[i] + 0.5 * alpha * dt / (dx**2) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Full step for advection (assuming a simple upwind scheme for demonstration)
    # Here, we assume a constant advection speed c, which needs to be defined
    c = 1.0  # Example advection speed
    for i in range(1, n-1):
        u_new[i] = u_new[i] - c * dt / dx * (u_new[i] - u_new[i-1])
    
    # Second half-step for diffusion
    for i in range(1, n-1):
        u_new[i] = u_new[i] + 0.5 * alpha * dt / (dx**2) * (u_new[i+1] - 2*u_new[i] + u_new[i-1])
    
    # Return the updated solution
    return u_new

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('40.2', 3)
target = targets[0]

u = np.array([-1,-1, -1, 0,1,2,3,4,5,6])
dt  = 0.1
dx  = 0.01
alpha = 0.5
assert np.allclose(Strang_splitting(u, dt, dx, alpha), target)
target = targets[1]

u = np.array([0,1,2,3,4,5,6])
dt  = 0.1
dx  = 0.1
alpha = 0.2
assert np.allclose(Strang_splitting(u, dt, dx, alpha), target)
target = targets[2]

u = np.array([0,1,2,4,6,8,0,1,23])
dt  = 0.01
dx  = 0.05
alpha = -0.2
assert np.allclose(Strang_splitting(u, dt, dx, alpha), target)
