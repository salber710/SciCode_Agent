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



# Background: Strang splitting is a numerical method used to solve partial differential equations (PDEs) that involve multiple processes, such as advection and diffusion. 
# It is a type of operator splitting method that allows for the separate treatment of different terms in the PDE. 
# The first order Strang splitting involves splitting the time evolution operator into two parts, typically representing different physical processes, and applying them sequentially over half time steps.
# For a diffusion equation, the Strang splitting can be applied by first evolving the system with the diffusion operator for half a time step, then applying any other operators (e.g., advection) for a full time step, and finally applying the diffusion operator again for another half time step.
# This method is particularly useful for problems where different processes can be computed more efficiently or accurately when treated separately.

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
    
    # Half time step for diffusion
    dt_half = dt / 2.0
    
    # Apply diffusion for half time step
    for i in range(1, n-1):
        u_new[i] = u[i] + alpha * dt_half / (dx**2) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Update the solution array
    u[:] = u_new[:]
    
    # Here, you would apply any other operators (e.g., advection) for a full time step
    # For this example, we assume no other operators are applied
    
    # Apply diffusion for another half time step
    for i in range(1, n-1):
        u_new[i] = u[i] + alpha * dt_half / (dx**2) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Update the solution array
    u[:] = u_new[:]
    
    return u


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
