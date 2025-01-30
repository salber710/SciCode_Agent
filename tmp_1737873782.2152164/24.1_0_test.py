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