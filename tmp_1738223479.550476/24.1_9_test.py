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


try:
    targets = process_hdf5_to_tuple('24.1', 3)
    target = targets[0]
    n_x = 10
    assert np.allclose(make_IC(n_x), target)

    target = targets[1]
    n_x = 100
    assert np.allclose(make_IC(n_x), target)

    target = targets[2]
    n_x = 30
    assert np.allclose(make_IC(n_x), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e