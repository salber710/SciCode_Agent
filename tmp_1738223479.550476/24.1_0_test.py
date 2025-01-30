from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The initial condition is specified as a piecewise function, and we need to compute cell-averaged values
# over each interval using numerical integration. For this purpose, we use the three-point Gauss quadrature rule,
# which is a method for approximating the integral of a function. The three-point Gauss quadrature rule is particularly
# efficient because it provides exact results for polynomials of degree up to 5. Given the piecewise nature of the function
# u_0, we need to integrate separately over the two intervals: (-π/2, 0] and (0, π/2). The Gauss quadrature rule uses
# specific weights and nodes within each interval to approximate the integral accurately.


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

    # Gauss quadrature weights and points for three-point rule (in reference interval [-1, 1])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    
    # Prepare the output array
    v = np.zeros(n-1)
    
    # Iterate over each cell to calculate the cell-averaged value
    for i in range(n-1):
        # Define the local interval for the current cell
        x_left = x_start + i * dx
        x_right = x_start + (i + 1) * dx
        
        # Map Gauss points from [-1, 1] to the current interval
        mapped_points = 0.5 * (x_right - x_left) * gauss_points + 0.5 * (x_left + x_right)
        
        # Evaluate the initial condition function at the mapped Gauss points
        u_values = np.where(mapped_points <= 0, np.sin(mapped_points) - 1, np.sin(mapped_points) + 1)
        
        # Compute the integral using the Gauss quadrature rule
        integral_value = 0.5 * (x_right - x_left) * np.sum(gauss_weights * u_values)
        
        # Compute the cell-averaged value
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