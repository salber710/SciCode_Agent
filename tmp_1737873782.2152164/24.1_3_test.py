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
    # Define the domain boundaries
    x_start = -np.pi / 2
    x_end = np.pi / 2
    
    # Calculate the cell width
    delta_x = (x_end - x_start) / (n - 1)
    
    # Initialize the cell-averaged values array
    v = np.zeros(n-1)
    
    # Gauss quadrature points and weights for three-point rule
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    # Function to evaluate u_0 at a given x
    def u_0(x):
        if -np.pi/2 < x <= 0:
            return np.sin(x) - 1
        elif 0 < x < np.pi/2:
            return np.sin(x) + 1
        else:
            return 0  # Out of bounds
    
    # Loop over each cell
    for i in range(n-1):
        # Calculate the cell boundaries
        x_left = x_start + i * delta_x
        x_right = x_left + delta_x
        
        # Perform Gauss quadrature over the cell
        cell_average = 0
        for gp, gw in zip(gauss_points, gauss_weights):
            # Convert Gauss point to the actual point in the cell
            x_gp = (x_right + x_left)/2 + (x_right - x_left)/2 * gp
            cell_average += gw * u_0(x_gp)
        
        # Scale by half the interval length (since transformation done)
        cell_average *= (x_right - x_left) / 2
        
        # Store the cell averaged value
        v[i] = cell_average
    
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