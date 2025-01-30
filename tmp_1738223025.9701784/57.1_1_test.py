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
    
    # Instead of using np.square or direct arithmetic, we use a different approach
    # Calculate x^2 using a for loop if x is a list, otherwise calculate it directly
    if isinstance(x, (list, np.ndarray)):
        x_squared = [xi * xi for xi in x]
    else:
        x_squared = x * x
    
    # Compute f(x) using the transformed potential and energy terms
    f_x = [2 * (v - En) for v in x_squared] if isinstance(x_squared, list) else 2 * (x_squared - En)
    
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