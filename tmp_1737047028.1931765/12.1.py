from scipy import integrate
from scipy import optimize
import numpy as np



# Background: 
# The Schrödinger equation for a hydrogen-like atom can be expressed in spherical coordinates. 
# For the radial part, the equation can be transformed into a form involving the radial wave function u(r), 
# where u(r) = r * R(r) and R(r) is the radial part of the wave function. 
# The transformed equation is u''(r) = f(r)u(r), where f(r) is a function that depends on the potential energy, 
# the angular momentum quantum number l, and the energy of the system.
# The potential energy term for a hydrogen-like atom is given by the Coulomb potential, 
# V(r) = -Ze^2 / (4πε₀r), where Z is the atomic number (Z=1 for hydrogen), e is the elementary charge, 
# and ε₀ is the permittivity of free space. 
# The effective potential also includes a centrifugal term due to the angular momentum, 
# which is l(l+1)ħ²/(2mr²), where ħ is the reduced Planck's constant and m is the electron mass.
# The function f(r) is derived from these terms and the given energy E.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J·s
    m_e = 9.10938356e-31  # Electron mass in kg
    e = 1.60217662e-19    # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    Z = 1  # Atomic number for hydrogen

    # Precompute constants
    hbar2_over_2m = hbar**2 / (2 * m_e)
    Ze2_over_4pi_epsilon0 = Z * e**2 / (4 * np.pi * epsilon_0)

    # Calculate f(r) for each r in r_grid
    f_r = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r == 0:
            # Avoid division by zero; handle r = 0 case separately if needed
            f_r[i] = 0
        else:
            # Effective potential term
            V_eff = -Ze2_over_4pi_epsilon0 / r + l * (l + 1) * hbar2_over_2m / r**2
            # f(r) = (2m/ħ²) * (V_eff - E)
            f_r[i] = (2 * m_e / hbar**2) * (V_eff - energy)

    return f_r


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
