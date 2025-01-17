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


from scicode.parse.parse import process_hdf5_to_tuple

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
