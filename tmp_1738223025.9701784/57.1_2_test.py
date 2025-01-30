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
    
    # Define a helper function to compute the potential term
    def potential_term(x_value):
        return x_value ** 2
    
    # Define a helper function to compute f(x) for a single x value
    def compute_f_single(x_value, energy):
        return 2 * (potential_term(x_value) - energy)
    
    # Use numpy for vectorized operations if x is an array
    if isinstance(x, (list, np.ndarray)):

        x_array = np.asarray(x)
        f_x_array = 2 * (potential_term(x_array) - En)
        return f_x_array
    else:
        # Compute directly for a single float value of x
        return compute_f_single(x, En)


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