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
    hbar = 1.0545718e-34  # Planck's constant / 2pi, in m^2 kg / s
    m_e = 9.10938356e-31  # mass of an electron, in kg
    e2 = 1.602176634e-19  # elementary charge squared, in C^2
    epsilon_0 = 8.854187817e-12  # vacuum permittivity, in F/m

    # Convert constants to atomic units where hbar = m_e = e^2/(4 * pi * epsilon_0) = 1
    # Compute the factor for the kinetic energy term
    kinetic_factor = 1 / (2 * hbar**2 / m_e)

    # Compute the potential energy term factor
    potential_factor = (e2 / (4 * 3.141592653589793 * epsilon_0))

    # Calculate f(r) using a different approach by splitting terms explicitly
    f_r = []
    for r in r_grid:
        if r != 0:  # Avoid division by zero
            # Kinetic term: -2 * energy
            kinetic_term = -2 * energy * kinetic_factor

            # Potential energy: -2 * (1/r)
            potential_term = -2 * (potential_factor / r)

            # Centrifugal term: l * (l + 1) / r^2
            centrifugal_term = l * (l + 1) / (r**2)

            # Compute f(r) as a sum of terms
            f_r_value = kinetic_term + potential_term + centrifugal_term
        else:
            # Handle r=0 case separately (could be set to a large value)
            f_r_value = float('inf')

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