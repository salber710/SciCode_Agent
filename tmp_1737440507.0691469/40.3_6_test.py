import numpy as np

# Background: The second order derivative of a function can be approximated using finite difference methods.
# A centered symmetric scheme provides a second order accurate approximation of the second derivative. 
# For a uniform grid with spacing `dx`, the second order centered difference formula for the second derivative 
# is given by (u[i+1] - 2*u[i] + u[i-1]) / (dx**2). This formula requires values from both sides of the target 
# point, making it a centered scheme. Ghost cells are used to handle boundary conditions, where the ghost cell 
# value is set equal to the nearest real cell value, effectively creating a Neumann boundary condition 
# (zero derivative at the boundary).

def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''

    # Handle boundary conditions using ghost cells
    if target == 0:
        # Use nearest cell value for the left ghost cell
        left = u[0]
    else:
        left = u[target - 1]

    if target == len(u) - 1:
        # Use nearest cell value for the right ghost cell
        right = u[len(u) - 1]
    else:
        right = u[target + 1]

    # Calculate second order derivative using centered difference
    deriv = (right - 2 * u[target] + left) / (dx ** 2)

    return deriv


# Background: Strang splitting is a numerical method used to solve partial differential equations by splitting the problem 
# into simpler sub-problems that can be solved sequentially. It is particularly useful for solving problems where the equation 
# can be decomposed into different parts, such as diffusion and advection processes. Strang splitting involves solving one 
# part of the equation for half a time step, then solving the other part for a full time step, and finally solving the first 
# part again for another half time step. For a diffusion process described by the equation ∂u/∂t = α * ∂²u/∂x², Strang 
# splitting can be applied by using the second order derivative function and the given diffusive coefficient.


def Strang_splitting(u, dt, dx, alpha):
    '''Inputs:
    u : solution, array of float
    dt: time interval, float
    dx: spatial interval, float
    alpha: diffusive coefficient, float
    Outputs:
    u : solution, array of float
    '''
    
    # Number of spatial points
    N = len(u)
    
    # Create a copy of u to store intermediate results
    u_half = np.copy(u)
    
    # First half-step: Diffusion using second order derivatives
    for i in range(N):
        second_derivative = second_diff(i, u, dx)
        u_half[i] = u[i] + 0.5 * dt * alpha * second_derivative
    
    # Full step: Advection (or any other part of the equation, but assuming zero here)
    # Since full step for advection or missing physical process is zero in this context, we skip this.
    
    # Second half-step: Diffusion again using second order derivatives
    u_check = np.copy(u_half)
    for i in range(N):
        second_derivative = second_diff(i, u_half, dx)
        u_check[i] = u_half[i] + 0.5 * dt * alpha * second_derivative
    
    return u_check



# Background: 
# The diffusion-reaction equation can be represented as ∂u/∂t = α * ∂²u/∂x² + R(u), where α is the diffusion 
# coefficient, and R(u) is the reaction term. In this context, we employ a Strang splitting scheme to separate 
# the diffusion and reaction processes. We use a second order spatial differentiation operator for the diffusion 
# part, as previously implemented. For time integration, we utilize the first order forward Euler method. 
# The CFL (Courant-Friedrichs-Lewy) condition is important for determining the stability of the numerical scheme, 
# typically related to the time step size in relation to the spatial discretization and wave speed.
# The reaction term is not given explicitly; however, for this implementation, we assume a simple linear reaction 
# term R(u) = ru, which can be adjusted as needed.


def solve(CFL, T, dt, alpha):
    '''Inputs:
    CFL : Courant-Friedrichs-Lewy condition number
    T   : Max time, float
    dt  : Time interval, float
    alpha : diffusive coefficient , float
    Outputs:
    u   : solution, array of float
    '''
    
    # Spatial domain
    L = 1.0  # Assuming unit length for simplicity
    dx = np.sqrt(dt * alpha / CFL)  # Derive dx from CFL condition

    # Number of grid points
    N = int(L / dx) + 1
    
    # Initial condition
    u = np.ones(N)  # Initial condition can be adjusted as needed
    r = 0.1  # Reaction rate coefficient

    # Time loop
    time = 0.0
    while time < T:
        u = Strang_splitting(u, dt, dx, alpha)
        
        # Apply reaction term using forward Euler
        u = u + dt * r * u
        
        # Update time
        time += dt
        
    return u

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
