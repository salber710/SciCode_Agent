from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np




def f_Schrod(energy, l, r_grid):
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J*s
    m = 9.10938356e-31    # Mass of electron in kg
    e = 1.602176634e-19   # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    Z = 1  # Atomic number for hydrogen

    # Precompute constants
    factor = 2 * m / hbar**2
    V_prefactor = -Z * e**2 / (4 * np.pi * epsilon_0)

    # Calculate f(r) using numpy operations for efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        V_r = V_prefactor / r_grid
        centrifugal_term = l * (l + 1) / r_grid**2
        f_r = factor * (V_r - energy) + centrifugal_term
        f_r[np.isinf(f_r)] = 0  # Handle infinities due to division by zero

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