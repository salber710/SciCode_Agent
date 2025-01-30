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

    # Gauss quadrature weights and nodes for three-point rule (in reference interval [-1, 1])
    gauss_weights = [5/9, 8/9, 5/9]
    gauss_nodes = [-np.sqrt(3/5), 0, np.sqrt(3/5)]

    # Prepare the output array
    v = []

    # Iterate over each cell to calculate the cell-averaged value
    for i in range(n-1):
        # Define the center of the current cell
        x_center = x_start + (i + 0.5) * dx

        # Map Gauss nodes from [-1, 1] to the current cell
        local_points = [(x_center + 0.5 * dx * node) for node in gauss_nodes]

        # Evaluate the initial condition function at the local Gauss points
        u_values = [np.sin(point) - 1 if point <= 0 else np.sin(point) + 1 for point in local_points]

        # Compute the integral using the Gauss quadrature rule
        integral_value = sum(weight * u_val for weight, u_val in zip(gauss_weights, u_values)) * (dx / 2)

        # Compute the cell-averaged value and append to the list
        v.append(integral_value / dx)

    return np.array(v)


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