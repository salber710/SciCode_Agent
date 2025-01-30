from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np



def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of floats
    Output
    f_r: a 1D array of floats 
    '''

    # Using a symbolic approach for the atomic unit constants
    # Constants are implicitly set as hbar = m_e = e^2/(4 * pi * epsilon_0) = 1 in atomic units
    # Define the function f(r) using a completely different breakdown of terms
    # Factor out certain terms to create a distinct implementation

    # Initialize the output array
    f_r = []

    # Iterate over each radius in the grid
    for r in r_grid:
        if r != 0:
            # Alternative rearrangement of the equation:
            # Group terms differently: -(2 * energy + l * (l + 1)/r^2) + 2/r
            # Here, split the potential and centrifugal terms distinctly
            energy_term = -2 * energy
            centrifugal_term = -(l * (l + 1)) / (r**2)
            potential_term = 2 / r

            # Calculate f(r) as a combination of the above terms
            f_r_value = energy_term + centrifugal_term + potential_term
        else:
            # Handle the r=0 case separately by assigning a very large value
            f_r_value = 1e10  # Assign a large number instead of infinity

        # Append the calculated value to the result list
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