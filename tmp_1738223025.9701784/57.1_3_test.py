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
    
    # Check if x is a single float or a list/array
    if isinstance(x, (list, tuple)):
        # Use list comprehension with map to apply the transformation
        f_x = list(map(lambda xi: 2 * (xi * xi - En), x))
    elif isinstance(x, np.ndarray):
        # Use numpy's vectorized approach
        f_x = 2 * (x ** 2 - En)
    else:
        # Direct computation for a single float value
        f_x = 2 * (x ** 2 - En)
    
    return f_x


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