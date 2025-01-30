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


    # Constants in atomic units: hbar = m_e = e^2/(4 * pi * epsilon_0) = 1

    # Use a unique approach by applying a different sequence of operations
    # First calculate the inverse of r_grid to handle division separately
    inv_r_grid = np.reciprocal(r_grid, where=r_grid!=0)

    # Calculate each term separately and combine them uniquely
    energy_term = -2 * energy
    potential_term = -2 * inv_r_grid
    centrifugal_term = l * (l + 1) * inv_r_grid**2

    # Combine the terms in a distinct order
    f_r = energy_term + centrifugal_term + potential_term

    # Handle the singularity at r = 0 by assigning a large finite value
    f_r[r_grid == 0] = 1e8  # Assign a large number for r = 0

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