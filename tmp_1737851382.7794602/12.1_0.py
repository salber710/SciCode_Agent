from scipy import integrate
from scipy import optimize
import numpy as np



# Background: 
# The Schrödinger equation for a hydrogen-like atom can be expressed in spherical coordinates. 
# For a central potential, the radial part of the Schrödinger equation can be transformed into a form 
# that resembles a one-dimensional Schrödinger equation. This is done by substituting the wave function 
# with u(r) = r * R(r), where R(r) is the radial wave function. The resulting equation is:
# u''(r) = f(r) * u(r), where f(r) is a function of the potential energy, kinetic energy, and the 
# angular momentum quantum number l. 
# For the hydrogen atom, the potential energy is given by the Coulomb potential, and the effective 
# potential includes a centrifugal term due to the angular momentum. 
# The function f(r) can be derived from the radial part of the Schrödinger equation:
# f(r) = -2m/ħ^2 * (E + Z*e^2/(4πε₀r) - l(l+1)ħ^2/(2mr^2))
# In atomic units, where ħ = 1, m = 1, e = 1, and 4πε₀ = 1, this simplifies to:
# f(r) = -2 * (energy + 1/r - l*(l+1)/(2*r^2))
# Here, Z is set to 1 for the hydrogen atom.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Calculate f(r) for each r in the radial grid
    f_r = -2 * (energy + 1/r_grid - l*(l+1)/(2*r_grid**2))
    
    return f_r

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
