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
    # Constants
    h_bar = 1.0545718e-34  # Planck's constant over 2π in J·s
    m_e = 9.10938356e-31   # Mass of an electron in kg
    e_charge = 1.60217662e-19  # Elementary charge in coulombs
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

    # Reduced mass for hydrogen-like atom (assuming Z=1 and nucleus is a proton)
    mu = m_e  # For hydrogen, the reduced mass is approximately the electron mass

    # Pre-computed constant factor in the equation
    factor = -h_bar**2 / (2 * mu)

    # Initialize the output array
    f_r = np.zeros_like(r_grid)

    # Calculate f(r) for each radius in r_grid
    for i, r in enumerate(r_grid):
        if r != 0:  # Avoid division by zero
            # Coulomb potential term
            V_r = -e_charge**2 / (4 * np.pi * epsilon_0 * r)
        
            # Centrifugal term due to angular momentum
            centrifugal_term = (l * (l + 1) * h_bar**2) / (2 * mu * r**2)

            # Calculate f(r) using the relation from the modified Schrödinger equation
            f_r[i] = (2 * mu / h_bar**2) * (V_r - energy + centrifugal_term)
        else:
            # Handle the r=0 case separately if needed
            f_r[i] = np.inf  # or some large value to indicate singularity

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