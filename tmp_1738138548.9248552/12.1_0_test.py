from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np



# Background: 
# The Schrödinger equation for a hydrogen-like atom can be expressed in spherical coordinates. 
# For the radial part, the equation can be transformed into a form involving the radial wave function u(r), 
# which is related to the original wave function by u(r) = r * R(r). The radial equation becomes:
# u''(r) = f(r) * u(r), where f(r) is a function that depends on the potential energy, 
# the angular momentum quantum number l, and the energy of the system.
# The potential energy term for a hydrogen-like atom is given by the Coulomb potential, 
# V(r) = -Z * e^2 / (4 * pi * epsilon_0 * r), where Z is the atomic number (Z=1 for hydrogen).
# The effective potential also includes a centrifugal term due to the angular momentum, 
# which is l * (l + 1) * hbar^2 / (2 * m * r^2).
# The function f(r) can be derived from the radial Schrödinger equation as:
# f(r) = (2 * m / hbar^2) * (V(r) - E) + l * (l + 1) / r^2,
# where E is the energy of the system, m is the mass of the electron, and hbar is the reduced Planck's constant.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J*s
    m = 9.10938356e-31    # Mass of electron in kg
    e = 1.602176634e-19   # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    Z = 1  # Atomic number for hydrogen

    # Precompute constants
    factor = 2 * m / hbar**2
    V_prefactor = -Z * e**2 / (4 * np.pi * epsilon_0)

    # Calculate f(r) for each r in r_grid
    f_r = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r == 0:
            # Avoid division by zero at r = 0
            f_r[i] = 0
        else:
            V_r = V_prefactor / r
            centrifugal_term = l * (l + 1) / r**2
            f_r[i] = factor * (V_r - energy) + centrifugal_term

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