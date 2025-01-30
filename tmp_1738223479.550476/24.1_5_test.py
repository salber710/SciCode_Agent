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

    # Gauss quadrature nodes and weights for three-point rule (in reference interval [-1, 1])
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])

    # Prepare the output array
    v = np.zeros(n-1)

    # Iterate over each cell to calculate the cell-averaged value
    for i in range(n-1):
        # Left boundary of the current cell
        x_left = x_start + i * dx
        x_right = x_left + dx

        # Map Gauss points from [-1, 1] to the current interval
        mapped_points = (x_right - x_left) / 2 * gauss_points + (x_right + x_left) / 2

        # Evaluate the initial condition function at the mapped Gauss points
        u_values = np.array([np.sin(x) - 1 if x <= 0 else np.sin(x) + 1 for x in mapped_points])

        # Compute the integral using the Gauss quadrature rule
        integral_value = np.sum(gauss_weights * u_values) * (x_right - x_left) / 2

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