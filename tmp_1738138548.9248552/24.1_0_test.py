from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: 
# The task is to compute the initial condition as a cell-averaged approximation using the three-point Gauss quadrature rule.
# Gauss quadrature is a numerical integration method that approximates the integral of a function. The three-point Gauss quadrature
# rule uses three specific points and weights to approximate the integral over an interval. For the interval [-1, 1], the points
# are x1 = -sqrt(3/5), x2 = 0, x3 = sqrt(3/5) with weights w1 = w3 = 5/9 and w2 = 8/9. To apply this to an arbitrary interval [a, b],
# we use a change of variables to map the interval [-1, 1] to [a, b].
# The initial condition function u_0 is piecewise defined, and we need to compute the cell-averaged values over each subinterval
# of the domain divided by the number of vertices n. The domain is [-pi/2, pi/2], and the function changes at x = 0.


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
        # Calculate the left and right endpoints of the cell
        a = x_start + i * dx
        b = a + dx
        
        # Map Gauss points from [-1, 1] to [a, b]
        mapped_points = 0.5 * (b - a) * gauss_points + 0.5 * (b + a)
        
        # Evaluate the function at the mapped points
        f_values = np.where(mapped_points <= 0, np.sin(mapped_points) - 1, np.sin(mapped_points) + 1)
        
        # Compute the integral using Gauss quadrature
        integral = 0.5 * (b - a) * np.sum(gauss_weights * f_values)
        
        # Compute the cell-averaged value
        v[i] = integral / dx
    
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