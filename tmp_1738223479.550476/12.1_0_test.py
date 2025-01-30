from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np



# Background: In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes over time. 
# For a hydrogen-like atom, the time-independent Schrödinger equation can be expressed in spherical coordinates, separating the radial 
# and angular components. In this form, the radial part of the Schrödinger equation involves the angular momentum quantum number `l` 
# and can be rewritten using the function `u(r) = r * R(r)` where `R(r)` is the radial wave function. This leads to a second-order 
# differential equation of the form `u''(r) = f(r)u(r)`, where `f(r)` needs to be determined. The potential energy term includes 
# the Coulomb potential for a charge `Z` nucleus (here Z=1 for hydrogen). The function `f(r)` includes terms for kinetic energy, 
# potential energy, and the centrifugal barrier due to angular momentum.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''

    # Constants
    hbar = 1.0545718e-34  # Planck's constant / 2pi, in m^2 kg / s
    m_e = 9.10938356e-31  # mass of an electron, in kg
    e2 = 1.602176634e-19  # elementary charge squared, in C^2
    epsilon_0 = 8.854187817e-12  # vacuum permittivity, in F/m

    # Convert constants to atomic units where hbar = m_e = e^2/(4 * pi * epsilon_0) = 1
    # This simplifies calculations as hbar^2/(2m) becomes 1/2 and Z e^2/(4 * pi * epsilon_0) becomes 1/r for Z = 1
    # Therefore, in atomic units, the equation becomes:
    # f(r) = -2 * (energy + 1/r) + l * (l + 1) / r^2

    # Calculate f(r) for each r in r_grid
    f_r = -2 * (energy + 1/r_grid) + l * (l + 1) / r_grid**2

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