import numpy as np



# Background: 
# The task is to compute the initial condition as a cell-averaged approximation using the three-point Gauss quadrature rule.
# Gauss quadrature is a numerical integration method that approximates the integral of a function. The three-point Gauss quadrature
# rule uses three specific points and weights to approximate the integral over an interval. For the interval [-1, 1], the points
# are x1 = -sqrt(3/5), x2 = 0, x3 = sqrt(3/5) with weights w1 = w3 = 5/9 and w2 = 8/9. To apply this rule to an arbitrary interval
# [a, b], we need to transform these points and weights accordingly.
# The initial condition function u_0 is piecewise defined over the interval [-pi/2, pi/2]. The task is to compute the cell-averaged
# values over n-1 cells, where each cell is defined by the vertices of the grid.


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
        # Define the cell boundaries
        a = x_start + i * dx
        b = a + dx
        
        # Transform Gauss points to the current cell [a, b]
        transformed_points = 0.5 * (b - a) * gauss_points + 0.5 * (b + a)
        
        # Evaluate the piecewise function at the transformed Gauss points
        integrand_values = np.where(transformed_points <= 0, 
                                    np.sin(transformed_points) - 1, 
                                    np.sin(transformed_points) + 1)
        
        # Compute the integral using Gauss quadrature
        integral = 0.5 * (b - a) * np.sum(gauss_weights * integrand_values)
        
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
