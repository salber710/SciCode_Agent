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


def Strang_splitting(u, dt, dx, alpha, c=1.0):
    '''Inputs:
    u : solution, array of float
    dt: time interval , float
    dx: spatial interval, float
    alpha: diffusive coefficient, float
    c: advection speed, float (default is 1.0)
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
    if c >= 0:
        for i in range(1, n):
            u_new[i] = u_new[i] - c * dt / dx * (u_new[i] - u_new[i-1])
    else:
        for i in range(n-2, -1, -1):
            u_new[i] = u_new[i] - c * dt / dx * (u_new[i+1] - u_new[i])
    
    # Second half-step for diffusion
    for i in range(1, n-1):
        u_new[i] = u_new[i] + 0.5 * alpha * dt / (dx**2) * (u_new[i+1] - 2*u_new[i] + u_new[i-1])
    
    # Boundary conditions are not updated by advection or diffusion steps, so set them to original values
    u_new[0] = u[0]
    u_new[-1] = u[-1]
    
    # Return the updated solution
    return u_new



# Background: The diffusion-reaction equation is a type of partial differential equation (PDE) that models processes involving both diffusion and reaction. 
# The diffusion term describes the spread of a quantity (e.g., heat, concentration) over space, while the reaction term accounts for local changes due to chemical reactions or other processes.
# To solve this equation numerically, we can use a combination of spatial discretization and time-stepping methods. 
# The second order spatial differentiation operator provides a more accurate approximation of the spatial derivatives, which is crucial for capturing the diffusion process accurately.
# The Strang splitting scheme allows us to handle the diffusion and reaction processes separately, improving the stability and accuracy of the solution.
# The first order forward Euler method is a simple and explicit time-stepping scheme that updates the solution at each time step based on the current state.


def solve(CFL, T, dt, alpha):
    '''Inputs:
    CFL : Courant-Friedrichs-Lewy condition number
    T   : Max time, float
    dt  : Time interval, float
    alpha : diffusive coefficient , float
    Outputs:
    u   : solution, array of float
    '''
    # Define the spatial domain
    L = 1.0  # Length of the domain
    dx = np.sqrt(alpha * dt / CFL)  # Spatial step size based on CFL condition
    nx = int(L / dx) + 1  # Number of spatial points
    x = np.linspace(0, L, nx)  # Spatial grid

    # Initial condition: assume a Gaussian distribution
    u = np.exp(-100 * (x - 0.5)**2)

    # Time-stepping loop
    t = 0.0
    while t < T:
        # Apply Strang splitting for diffusion-reaction
        u = Strang_splitting(u, dt, dx, alpha)

        # Update time
        t += dt

    return u

def Strang_splitting(u, dt, dx, alpha):
    '''Strang splitting for diffusion-reaction equation.'''
    # Number of spatial points
    n = len(u)
    
    # Create a copy of u to store the updated solution
    u_new = np.copy(u)
    
    # First half-step for diffusion
    for i in range(1, n-1):
        u_new[i] = u[i] + 0.5 * alpha * dt / (dx**2) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Reaction step (for demonstration, assume a simple linear reaction term)
    reaction_rate = 0.1  # Example reaction rate
    u_new = u_new + dt * reaction_rate * u_new
    
    # Second half-step for diffusion
    for i in range(1, n-1):
        u_new[i] = u_new[i] + 0.5 * alpha * dt / (dx**2) * (u_new[i+1] - 2*u_new[i] + u_new[i-1])
    
    # Boundary conditions are not updated by advection or diffusion steps, so set them to original values
    u_new[0] = u[0]
    u_new[-1] = u[-1]
    
    return u_new

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('40.3', 3)
target = targets[0]

CFL = 0.2
T   = 0.1
dt  = 0.01
alpha = 0.1
assert np.allclose(solve(CFL, T, dt, alpha), target)
target = targets[1]

CFL = 0.3
T   = 0.3
dt  = 0.05
alpha = 0.05
assert np.allclose(solve(CFL, T, dt, alpha), target)
target = targets[2]

CFL = 0.1
T   = 0.5
dt  = 0.01
alpha = 0.2
assert np.allclose(solve(CFL, T, dt, alpha), target)
