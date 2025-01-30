from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def make_IC(n):
    '''The function computes the initial condition mentioned above
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n
    '''

    # Define the domain from -pi/2 to pi/2
    x_min = -np.pi / 2
    x_max = np.pi / 2

    # Calculate the width of each cell
    dx = (x_max - x_min) / n

    # Precompute the Gauss quadrature points and weights for 3-point rule
    # Points are in the interval [-1, 1], so we need to scale them to the interval of each cell
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])

    # Scale the points for each cell
    scaled_points = 0.5 * dx * gauss_points + 0.5 * dx

    # Initialize the array to store cell-averaged values
    v = np.zeros(n)

    # Loop over each cell
    for i in range(n):
        # Compute the left and right boundaries of the cell
        x_left = x_min + i * dx
        x_right = x_left + dx

        # Transform Gauss points to the actual x values within the cell
        x_points = x_left + scaled_points

        # Evaluate the initial condition function at the Gauss points
        u_values = np.where(
            x_points <= 0,
            np.sin(x_points) - 1,
            np.sin(x_points) + 1
        )

        # Compute the integral approximation over the cell
        integral = np.sum(gauss_weights * u_values) * 0.5 * dx

        # Store the cell-averaged value
        v[i] = integral / dx

    return v


def LaxF(uL, uR):
    '''This function computes Lax-Friedrichs numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    
    # Compute the fluxes at the left and right states using the flux function F(u) = 0.5 * u^2
    FL = 0.5 * uL**2
    FR = 0.5 * uR**2

    # Calculate the maximum wave speed for the Lax-Friedrichs stability parameter
    alpha_LF = max(abs(uL), abs(uR))

    # Compute the Lax-Friedrichs numerical flux
    flux = 0.5 * (FL + FR) - 0.5 * alpha_LF * (uR - uL)

    return flux



def solve(n_x, n_t, T):
    '''Inputs:
    n_x : number of spatial grids, Integer
    n_t : number of temporal grids, Integer
    T   : final time, float
    Outputs
    u1   : solution vector, 1d array of size n_x-1
    '''
    
    # Importing necessary functions


    # Initialize spatial domain
    x_min = -pi / 2
    x_max = pi / 2
    dx = (x_max - x_min) / n_x

    # Initialize time step
    dt = T / n_t

    # Initialize solution array
    u = make_IC(n_x)

    # Time-stepping loop
    for n in range(n_t):
        # Create a new array for the updated solution
        u_new = zeros(n_x)
        
        # Compute fluxes at each interface
        for i in range(1, n_x - 1):
            # Compute the flux using Lax-Friedrichs at the left and right interfaces
            flux_left = LaxF(u[i-1], u[i])
            flux_right = LaxF(u[i], u[i+1])
            
            # Update the solution for the current cell using finite volume method
            u_new[i] = u[i] - dt/dx * (flux_right - flux_left)
        
        # Apply free boundary conditions (zero gradient)
        u_new[0] = u[0]  # Free boundary at left
        u_new[-1] = u[-1]  # Free boundary at right

        # Update solution
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