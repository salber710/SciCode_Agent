from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def make_IC(n):
    x_start = -np.pi / 2
    x_end = np.pi / 2
    dx = (x_end - x_start) / (n - 1)
    
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    v = np.zeros(n - 1)
    
    for i in range(n - 1):
        a = x_start + i * dx
        b = a + dx
        mapped_points = 0.5 * (b - a) * gauss_points + 0.5 * (b + a)
        
        # Evaluate the function using numpy.piecewise
        f_values = np.piecewise(mapped_points, [mapped_points <= 0, mapped_points > 0],
                                [lambda x: np.sin(x) - 1, lambda x: np.sin(x) + 1])
        
        # Compute the integral using Gauss quadrature
        integral = np.dot(gauss_weights, f_values) * 0.5 * (b - a)
        
        # Compute the cell-averaged value
        v[i] = integral / dx
    
    return v


def LaxF(uL, uR):
    '''This function computes Lax-Friedrich numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Calculate alpha_LF using the logarithm of the sum of the exponentials of the absolute values of uL and uR

    alpha_LF = math.log(math.exp(abs(uL)) + math.exp(abs(uR)))

    # Compute the Lax-Friedrichs flux with a logarithmic-exponential dissipation term
    flux = 0.5 * (uL + uR) - 0.5 * alpha_LF * (uR - uL)

    return flux



# Background: The 1D Burgers' equation is a fundamental partial differential equation from fluid mechanics. 
# It is often used as a simplified model for various types of wave phenomena. In this problem, we are solving 
# the Burgers' equation using a finite volume method with a first-order Euler time-stepping scheme. 
# The finite volume method involves dividing the spatial domain into discrete volumes and applying conservation 
# laws to each volume. The Lax-Friedrichs flux is used to approximate the flux at the boundaries of these volumes, 
# incorporating a numerical dissipation term to stabilize the solution. Free boundary conditions imply that the 
# solution at the boundaries of the domain is not influenced by any external conditions, allowing the solution 
# to evolve naturally based on the internal dynamics.


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

    # Initialize the solution vector using the initial condition function
    u = make_IC(n_x)

    # Time-stepping loop
    for _ in range(n_t):
        u_new = np.zeros_like(u)
        
        # Apply free boundary conditions
        uL = np.concatenate(([u[0]], u))
        uR = np.concatenate((u, [u[-1]]))
        
        # Compute the fluxes using Lax-Friedrichs method
        fluxes = np.array([LaxF(uL[i], uR[i]) for i in range(n_x - 1)])
        
        # Update the solution using the finite volume method
        for i in range(1, n_x - 2):
            u_new[i] = u[i] - dt/dx * (fluxes[i] - fluxes[i-1])
        
        # Update the solution for the next time step
        u = u_new

    return u


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e