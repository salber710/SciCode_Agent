import numpy as np

# Background: 
# The task is to compute the initial condition as a cell-averaged approximation using the three-point Gauss quadrature rule.
# Gauss quadrature is a numerical integration method that approximates the integral of a function. The three-point Gauss quadrature
# rule uses three specific points and weights to approximate the integral over an interval. For the interval [-1, 1], the points
# are x1 = -sqrt(3/5), x2 = 0, x3 = sqrt(3/5) with weights w1 = w3 = 5/9 and w2 = 8/9. To apply this to an arbitrary interval [a, b],
# we transform the points and weights accordingly. The initial condition function u_0 is piecewise defined, and we need to compute
# the cell-averaged values over each subinterval of the domain [-pi/2, pi/2] divided into n-1 cells.


def make_IC(n):
    '''The function computes the initial condition mentioned above
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n-1
    '''
    # Define the domain
    x_start = -np.pi / 2
    x_end = np.pi / 2
    
    # Calculate the cell width
    dx = (x_end - x_start) / (n - 1)
    
    # Gauss quadrature points and weights for the interval [-1, 1]
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    # Initialize the array for cell-averaged values
    v = np.zeros(n - 1)
    
    # Loop over each cell
    for i in range(n - 1):
        # Calculate the left and right endpoints of the current cell
        x_left = x_start + i * dx
        x_right = x_left + dx
        
        # Transform Gauss points to the current cell
        x_gauss = 0.5 * (x_right - x_left) * gauss_points + 0.5 * (x_right + x_left)
        
        # Evaluate the initial condition at the Gauss points
        u_values = np.where(x_gauss <= 0, np.sin(x_gauss) - 1, np.sin(x_gauss) + 1)
        
        # Compute the integral using Gauss quadrature
        integral = np.sum(gauss_weights * u_values) * 0.5 * (x_right - x_left)
        
        # Compute the cell-averaged value
        v[i] = integral / dx
    
    return v


# Background: The Lax-Friedrichs method is a numerical technique used to solve hyperbolic partial differential equations.
# It is a type of finite volume method that approximates the flux at the interface between two cells. The Lax-Friedrichs
# flux is computed using the average of the fluxes from the left and right states, with a stabilization term that involves
# the maximum wave speed, denoted as alpha_LF. This stabilization term helps to ensure numerical stability by accounting
# for the maximum speed at which information can propagate through the system. The formula for the Lax-Friedrichs flux is:
# 
# flux = 0.5 * (f(uL) + f(uR)) - 0.5 * alpha_LF * (uR - uL)
#
# where f(u) is the flux function, which in the simplest linear case can be taken as f(u) = u. The parameter alpha_LF
# is chosen as the maximum wave speed in the system, which can be problem-specific.


def LaxF(uL, uR):
    '''This function computes Lax-Friedrichs numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Define the maximum wave speed, alpha_LF
    # For a simple linear advection problem, this can be the maximum absolute value of the wave speed.
    # Here, we assume a wave speed of 1 for simplicity, but this should be adjusted based on the problem.
    alpha_LF = 1.0
    
    # Compute the Lax-Friedrichs flux
    flux = 0.5 * (uL + uR) - 0.5 * alpha_LF * (uR - uL)
    
    return flux



# Background: The 1D Burgers' equation is a fundamental partial differential equation from fluid dynamics, often used as a simplified model for various types of wave propagation. It is given by the equation u_t + u * u_x = 0, where u is the velocity field, t is time, and x is the spatial coordinate. In this problem, we solve the Burgers' equation using a finite volume method with a first-order Euler time-stepping scheme. The finite volume method involves dividing the spatial domain into discrete cells and computing the fluxes at the cell interfaces. The Lax-Friedrichs method is used to compute these fluxes, providing numerical stability. The initial condition is provided by the make_IC function, and the Lax-Friedrichs flux is computed using the LaxF function. Free boundary conditions imply that the solution at the boundaries is not influenced by any external conditions, allowing the solution to evolve naturally based on the internal dynamics.

def solve(n_x, n_t, T):
    '''Inputs:
    n_x : number of spatial grids, Integer
    n_t : number of temporal grids, Integer
    T   : final time, float
    Outputs
    u1   : solution vector, 1d array of size n_x-1
    '''

    
    # Initialize the spatial grid and time step
    x_start = -np.pi / 2
    x_end = np.pi / 2
    dx = (x_end - x_start) / (n_x - 1)
    dt = T / n_t
    
    # Initialize the solution vector using the initial condition
    u = make_IC(n_x)
    
    # Time-stepping loop
    for _ in range(n_t):
        # Create a new array to store the updated solution
        u_new = np.zeros_like(u)
        
        # Update the solution using the Lax-Friedrichs flux
        for i in range(1, n_x - 2):
            # Compute the fluxes at the interfaces
            flux_left = LaxF(u[i-1], u[i])
            flux_right = LaxF(u[i], u[i+1])
            
            # Update the solution using the finite volume method
            u_new[i] = u[i] - dt/dx * (flux_right - flux_left)
        
        # Apply free boundary conditions (no update needed at boundaries)
        u_new[0] = u[0]
        u_new[-1] = u[-1]
        
        # Update the solution for the next time step
        u = u_new
    
    return u


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('24.3', 3)
target = targets[0]

n_x = 31
n_t = 31
T = 1
assert np.allclose(solve(n_x,n_t,T), target)
target = targets[1]

n_x = 21
n_t = 51
T = 2
assert np.allclose(solve(n_x,n_t,T), target)
target = targets[2]

n_x = 11
n_t = 11
T = 1
assert np.allclose(solve(n_x,n_t,T), target)
