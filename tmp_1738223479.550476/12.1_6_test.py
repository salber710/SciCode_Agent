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

    # Using a different approach by combining terms into a singular expression
    # Constants are in atomic units, where hbar = m_e = e^2/(4 * pi * epsilon_0) = 1
    # Rearrange and combine terms in a novel manner: f(r) = (-2 * (energy + 1/r)) + l*(l+1)/r**2

    # Initialize the output array using list comprehension for a concise structure
    f_r = [
        -2 * (energy + (1 / r)) + l * (l + 1) / r**2 if r != 0 else float('nan')
        for r in r_grid
    ]

    # Replace the singularity at r = 0 with NaN to signify undefined behavior
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