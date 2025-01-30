from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np



def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''


    # Constants in atomic units
    # Use Z = 1 for hydrogen
    Z = 1

    # Define the function using a different mathematical expression
    # Rearrange terms and factor out common elements
    # f(r) = -2 * energy - (2/r) + l * (l + 1) / r^2

    # Handle the singularity at r = 0 by explicitly checking for zero
    f_r = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r != 0:
            f_r[i] = -2 * energy - (2 / r) + l * (l + 1) / r**2
        else:
            f_r[i] = float('inf')  # Assign infinity for r = 0 to avoid division by zero

    return f_r


try:
    targets = process_hdf5_to_tuple('12.1', 3)
    target = targets[0]
    assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)

    target = targets[1]
    assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)

    target = targets[2]
    assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e