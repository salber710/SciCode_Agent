from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def make_IC(n):
    '''The function computes the initial condition mentioned above
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n-1
    '''
    # Define the domain from -π/2 to π/2
    x_start = -np.pi / 2
    x_end = np.pi / 2

    # Calculate the cell width
    dx = (x_end - x_start) / n

    # Define the Gauss quadrature weights and nodes for three-point rule
    gauss_weights = [5/9, 8/9, 5/9]
    gauss_nodes = [-np.sqrt(3/5), 0, np.sqrt(3/5)]

    # Prepare the output array
    v = np.zeros(n-1)

    # Precompute the transformation coefficients
    mid_factor = dx / 2
    shift_factor = x_start + mid_factor

    # Iterate over each cell to calculate the cell-averaged value
    for i in range(n-1):
        # Define the mid-point of the current cell
        x_mid = shift_factor + i * dx
        
        # Transform Gauss nodes to the current cell
        mapped_points = x_mid + mid_factor * np.array(gauss_nodes)
        
        # Evaluate the initial condition function at the mapped Gauss points
        u_values = np.sin(mapped_points) + np.where(mapped_points <= 0, -1, 1)
        
        # Compute the integral using the Gauss quadrature rule
        integral_value = np.dot(gauss_weights, u_values) * mid_factor
        
        # Store the cell-averaged value
        v[i] = integral_value / dx

    return v



def LaxF(uL, uR):
    '''This function computes Lax-Friedrich numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Define the maximum wave speed (stability parameter)
    alpha_LF = 1.0
    
    # Calculate the average and difference terms
    average_flux = (uL + uR) / 2
    diff_term = (uR - uL) / 2
    
    # Calculate the Lax-Friedrichs numerical flux using a different approach
    flux = average_flux - alpha_LF * diff_term

    return flux


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e