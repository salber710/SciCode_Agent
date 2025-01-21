from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np



# Background: The Schrödinger equation in quantum mechanics describes how the quantum state of a physical system changes over time. For a particle in a central potential, the time-independent Schrödinger equation can be written in spherical coordinates. By separating variables, the radial part of the equation can be transformed into a form where the second derivative of a radial function u(r) with respect to the radius r is set equal to a function f(r) times u(r), i.e., u''(r) = f(r)u(r). For the hydrogen atom, Z=1, and the potential term involves the angular momentum quantum number l. The function f(r) incorporates the kinetic energy term, the potential energy term, and the centrifugal barrier term due to angular momentum. In this problem, we are given a grid of radii (r_grid), an energy value (energy), and an angular momentum quantum number (l), and we need to calculate the function f(r) for each radius in r_grid.


def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Constants
    Z = 1  # Atomic number for hydrogen
    hbar = 1.0545718e-34  # Planck's constant over 2*pi, in J*s
    m = 9.10938356e-31  # Electron mass in kg
    e = 1.60217662e-19  # Elementary charge in C
    epsilon_0 = 8.85418782e-12  # Vacuum permittivity in F/m
    
    # Precompute constants
    c1 = -hbar**2 / (2 * m)
    c2 = -Z * e**2 / (4 * np.pi * epsilon_0)
    
    # Calculate f(r) for each r in r_grid
    f_r = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r == 0:
            f_r[i] = 0  # Avoid division by zero, handle r=0 case separately if necessary
        else:
            # Kinetic term + Potential term + Centrifugal term - Energy
            kinetic_term = c1 * (l * (l + 1)) / r**2
            potential_term = c2 / r
            f_r[i] = (2 * m / hbar**2) * (potential_term + kinetic_term - energy)
    
    return f_r


try:
    targets = process_hdf5_to_tuple('12.1', 3)
    target = targets[0]
    assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)

    target = targets[1]
    assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)

    target = targets[2]
    assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)

