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
    
    # Define the domain
    x_start = -np.pi / 2
    x_end = np.pi / 2
    
    # Calculate the cell width
    dx = (x_end - x_start) / n
    
    # Gauss quadrature points and weights for three-point rule
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    # Initialize the cell-averaged values
    v = np.zeros(n-1)
    
    # Compute cell-averaged values using Gauss quadrature
    for i in range(n-1):
        # Calculate the start and end of the current cell
        x_left = x_start + i * dx
        x_right = x_left + dx
        
        # Change of variables for Gauss quadrature:
        # Transform from [-1, 1] to [x_left, x_right]
        mid_point = (x_left + x_right) / 2
        half_width = dx / 2
        
        # Integrate using Gauss quadrature
        integral_value = 0
        for gp, gw in zip(gauss_points, gauss_weights):
            # Compute the actual x value for the quadrature point
            x = mid_point + half_width * gp
            
            # Evaluate the initial condition function at this x
            if -np.pi/2 < x <= 0:
                u0_value = np.sin(x) - 1
            elif 0 < x < np.pi/2:
                u0_value = np.sin(x) + 1
            else:
                u0_value = 0  # In case of boundaries, although this shouldn't happen
            
            # Add the weighted contribution
            integral_value += gw * u0_value
        
        # Average over the cell
        v[i] = (half_width * integral_value) / dx
    
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