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
    
    # Use a dictionary to store potential energy values for a range of x if x is iterable
    def compute_fx(x_value):
        V_x = x_value ** 2  # Calculate potential energy V(x) = x^2
        return 2 * (V_x - En)

    # Check if x is iterable
    if hasattr(x, '__iter__'):
        # Use a dictionary comprehension for a different approach
        result_dict = {i: compute_fx(xi) for i, xi in enumerate(x)}
        # Extract the computed values and return them as a list
        return list(result_dict.values())
    else:
        # Directly compute for a single float value
        return compute_fx(x)


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