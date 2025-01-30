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
    domain = np.linspace(-np.pi / 2, np.pi / 2, n)
    
    # Calculate the cell width
    dx = (domain[-1] - domain[0]) / (n - 1)

    # Gauss quadrature weights and nodes for three-point rule (in reference interval [-1, 1])
    gauss_data = [(5/9, -np.sqrt(3/5)), (8/9, 0), (5/9, np.sqrt(3/5))]

    # Prepare the output array
    v = np.zeros(n-1)
    
    # Iterate over each cell to calculate the cell-averaged value
    for i in range(n-1):
        # Define the endpoints of the current cell
        x_left = domain[i]
        x_right = domain[i + 1]

        # Initialize integral value for the cell
        integral_value = 0.0

        # Compute the integral using the Gauss quadrature rule
        for weight, node in gauss_data:
            # Map node from reference interval [-1, 1] to the current cell
            mapped_point = 0.5 * (x_right - x_left) * node + 0.5 * (x_left + x_right)

            # Evaluate the initial condition at this point
            if mapped_point <= 0:
                u_value = np.sin(mapped_point) - 1
            else:
                u_value = np.sin(mapped_point) + 1

            # Add to the integral value for this cell
            integral_value += weight * u_value

        # Finalize the integral calculation for this cell
        integral_value *= 0.5 * (x_right - x_left)

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