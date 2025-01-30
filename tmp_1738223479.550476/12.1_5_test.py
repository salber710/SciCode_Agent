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

    # Use mathematical transformations to create a unique implementation
    # Constants in atomic units: hbar = m_e = e^2/(4 * pi * epsilon_0) = 1
    # Formula transformation: factor out terms in a unique way

    # Initialize the output array
    f_r = []

    # Iterate over each radius in the grid
    for r in r_grid:
        if r != 0:
            # Expression breakdown: combine terms differently
            # f(r) = -(energy + 1/r) * 2 + l * (l + 1) / r**2
            # This uses a distinct ordering and grouping of terms
            combined_potential_energy = 1 / r
            combined_kinetic_energy = energy

            combined_term = -2 * (combined_kinetic_energy + combined_potential_energy)
            centrifugal_term = l * (l + 1) / r**2

            # Compute f(r) by aggregating the terms
            f_r_value = combined_term + centrifugal_term
        else:
            # Handle singularity at r = 0 by assigning a large value
            f_r_value = 1e10  # Assign a large number to avoid division by zero

        f_r.append(f_r_value)

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