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
    
    # Define a function to compute the potential energy
    def potential_energy(x_val):
        return x_val * x_val
    
    # Define a function to compute f(x) for a single value of x
    def compute_fx(x_val):
        V_x = potential_energy(x_val)
        return 2 * (V_x - En)
    
    # Check if x is a single value or iterable and process accordingly
    if isinstance(x, (list, tuple)):
        # Use a generator expression to compute f(x) for each element and convert to list
        return list(compute_fx(xi) for xi in x)
    elif isinstance(x, np.ndarray):
        # Utilize numpy's vectorized operations to compute f(x) for an array
        return 2 * (x**2 - En)
    else:
        # Direct computation for a single scalar value
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