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
    hbar = 1.0  # Planck's constant over 2π
    m_e = 1.0   # Electron mass
    e = 1.0     # Elementary charge
    epsilon_0 = 1.0  # Permittivity of free space
    Z = 1       # Atomic number for hydrogen

    # Precompute constant terms
    V_prefactor = -Z * e**2 / (4 * np.pi * epsilon_0)  # Coulomb potential prefactor
    H_prefactor = -hbar**2 / (2 * m_e)  # Kinetic energy prefactor

    # Calculate f(r) for each radius in r_grid
    f_r = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r != 0:
            # Radial potential V(r)
            V_r = V_prefactor / r

            # Centrifugal potential term
            L_r = l * (l + 1) * hbar**2 / (2 * m_e * r**2)

            # Total effective potential
            U_eff = V_r + L_r

            # Calculate f(r) as per the rewritten form u''(r) = f(r)u(r)
            f_r[i] = (2 * m_e / hbar**2) * (U_eff - energy)
        else:
            # Handle the r=0 case to avoid division by zero
            # The centrifugal term is dominant so we set f(r) to a large value
            # or use a limiting case where necessary
            f_r[i] = float('inf')  # or some large number depending on the context

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