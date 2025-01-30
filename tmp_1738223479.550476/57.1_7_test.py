from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize



def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    
    # Define a function to calculate f(x) for a single value
    def calculate_fx(x_val, energy):
        V_x = x_val * x_val  # Calculate potential energy V(x) = x^2
        return 2 * (V_x - energy)
    
    # Implement a recursive approach to handle lists or arrays
    def recursive_fx(x_vals, energy):
        if isinstance(x_vals, (list, tuple, np.ndarray)):
            return [recursive_fx(xi, energy) for xi in x_vals]
        else:
            return calculate_fx(x_vals, energy)
    
    # Call the recursive function with the input x and energy En
    return recursive_fx(x, En)


try:
    targets = process_hdf5_to_tuple('57.1', 3)
    target = targets[0]
    assert np.allclose(f_x(np.linspace(-5, 5, 10), 1), target)

    target = targets[1]
    assert np.allclose(f_x(np.linspace(0, 5, 10), 1), target)

    target = targets[2]
    assert np.allclose(f_x(np.linspace(0, 5, 20), 2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e