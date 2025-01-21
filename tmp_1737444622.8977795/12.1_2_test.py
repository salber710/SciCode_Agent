from scipy import integrate
from scipy import optimize
import numpy as np



# Background: 
# The Schrodinger equation in spherical coordinates for a central potential can be separated into a radial part. 
# By making a substitution, we can transform the radial Schrodinger equation into a form that resembles a simple 
# harmonic oscillator differential equation. This is typically done by writing the radial wave function as 
# a product u(r) = r * R(r), where R(r) is the radial part of the wave function. 
# Thus, the radial equation becomes a second-order differential equation in terms of u(r):
# u''(r) = f(r)u(r).
# The function f(r) is derived from the Schrodinger equation as follows:
# f(r) = (l*(l+1))/(r^2) - (2*Z/r) - (2*energy), where l is the angular momentum quantum number,
# Z is the atomic number (Z = 1 for hydrogen), and energy is the given energy level.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    Z = 1  # Atomic number for hydrogen
    # Calculate f(r) using the provided formula
    f_r = (l * (l + 1)) / (r_grid ** 2) - (2 * Z / r_grid) - (2 * energy)
    
    return f_r

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
