import numpy as np

# Background: 
# In numerical analysis, the three-point Gauss quadrature rule is used to approximate the integral of a function.
# It is a specific case of Gaussian quadrature where the function is approximated using three points within each interval.
# This method is particularly useful for integrating polynomial functions exactly up to degree 5.
# In this context, we are using it to compute the cell-averaged values of an initial condition function `u_0`.
# The function `u_0` is piecewise defined, with different expressions for the interval (-π/2, 0] and (0, π/2).
# For each cell, we will evaluate the integral using the Gauss quadrature rule and then divide by the cell width
# to obtain the average value.


def make_IC(n):
    '''The function computes the initial condition mentioned above.
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n-1
    '''
    
    # Define the interval [-π/2, π/2] and calculate the cell width
    x0 = -np.pi / 2
    x1 = np.pi / 2
    h = (x1 - x0) / (n - 1)
    
    # Weights and abscissas for the 3-point Gauss quadrature rule
    weights = np.array([5/9, 8/9, 5/9])
    abscissas = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    
    # Initialize the array for cell-averaged values
    v = np.zeros(n - 1)
    
    # Loop over each cell
    for i in range(n - 1):
        # Calculate the local cell interval
        xi = x0 + i * h
        xi_plus_1 = x0 + (i + 1) * h
        
        # Calculate the mid-point and half the length of the cell
        mid_point = (xi + xi_plus_1) / 2
        half_length = (xi_plus_1 - xi) / 2
        
        # Initialize the integral for the current cell
        integral = 0
        
        # Apply the Gauss quadrature rule
        for j in range(3):
            # Transform the abscissas to the local cell coordinates
            x = mid_point + half_length * abscissas[j]
            
            # Evaluate the piecewise function u_0 at the transformed abscissa
            if x <= 0:
                u_0 = np.sin(x) - 1
            else:
                u_0 = np.sin(x) + 1
            
            # Add the weighted contribution to the integral
            integral += weights[j] * u_0
        
        # Calculate the cell-averaged value
        v[i] = integral * half_length  # Multiply by half_length to account for the integral transformation
    
    return v



# Background: The Lax-Friedrichs method is a numerical scheme used to solve hyperbolic partial differential equations.
# It calculates numerical fluxes at the interfaces of discretized cells. The method introduces numerical dissipation
# by averaging the flux values from adjacent cells and subtracting a term proportional to the difference between the
# cell-averaged values. This stabilizing term is scaled by the maximum wave speed, known as the Lax-Friedrichs stability
# parameter, α_LF. This method is useful for ensuring stability in the numerical solution by mitigating oscillations.


def LaxF(uL, uR):
    '''This function computes Lax-Friedrichs numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Compute the maximum wave speed, which can be assumed or calculated based on the problem context.
    # For a simple transport equation, this might be a constant or derived from the maximum derivative of the flux function.
    alpha_LF = 1.0  # Assuming a constant wave speed for this example

    # Compute the numerical flux using the Lax-Friedrichs scheme
    flux = 0.5 * (uL + uR) - 0.5 * alpha_LF * (uR - uL)

    return flux

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('24.2', 3)
target = targets[0]

v_i = 0
v_i1 = 1
assert np.allclose(LaxF(v_i,v_i1), target)
target = targets[1]

v_i = 3
v_i1 = 3
assert np.allclose(LaxF(v_i,v_i1), target)
target = targets[2]

v_i = 5
v_i1 = -5
assert np.allclose(LaxF(v_i,v_i1), target)
