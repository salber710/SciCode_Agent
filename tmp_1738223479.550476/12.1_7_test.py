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


    # Constants in atomic units where hbar = m_e = e^2/(4 * pi * epsilon_0) = 1

    # Define a different transformation of the equation
    # Utilize numpy for efficient computation and distinct handling of terms
    # Express f(r) in a novel rearrangement to ensure uniqueness

    # Calculate f(r) using numpy's vectorized operations
    f_r = np.zeros_like(r_grid)
    non_zero_indices = r_grid != 0  # Boolean array for non-zero r values
    
    # Handle only non-zero r values to avoid division by zero
    r_non_zero = r_grid[non_zero_indices]
    f_r[non_zero_indices] = (
        -(2 * energy + 2 * (1 / r_non_zero)) + (l**2 + l) / r_non_zero**2
    )

    # Assign a defined large value at r = 0 to handle the singularity
    f_r[~non_zero_indices] = 1e6  # Use a large number to denote undefined behavior

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