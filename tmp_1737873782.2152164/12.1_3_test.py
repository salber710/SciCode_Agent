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
    hbar = 1.0545718e-34  # Planck's constant over 2*pi, in J*s
    e = 1.602176634e-19  # Elementary charge, in C
    epsilon_0 = 8.8541878128e-12  # Vacuum permittivity, in C^2/(N*m^2)
    m_e = 9.10938356e-31  # Electron mass, in kg
    Z = 1  # Atomic number for hydrogen

    # Reduced mass (for hydrogen, it's approximately the electron mass)
    mu = m_e
    
    # Convert energy from atomic units to Joules if necessary
    # energy = energy * some_conversion_factor (if energy is not already in Joules)

    # Calculate potential energy term V(r) = -Z*e^2 / (4 * pi * epsilon_0 * r)
    V_r = -Z * e**2 / (4 * np.pi * epsilon_0 * r_grid)

    # Calculate the centrifugal term l*(l+1) * hbar^2 / (2 * mu * r^2)
    centrifugal_term = l * (l + 1) * hbar**2 / (2 * mu * r_grid**2)

    # Calculate the function f(r) = (2*mu/hbar^2) * (V(r) - E) + l*(l+1)/(r^2)
    f_r = (2 * mu / hbar**2) * (V_r - energy) + centrifugal_term
    
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