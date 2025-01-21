from scipy import integrate
from scipy import optimize
import numpy as np



# Background: 
# The problem involves transforming the radial part of the Schrödinger equation for a hydrogen-like atom.
# The given Schrödinger equation is in the form:
# (-ħ²/2m)∇²ψ(r) - (Ze²/4πε₀r)ψ(r) = Eψ(r)
# We are asked to rewrite this equation in a different form: u''(r) = f(r)u(r).
# In this radial form, u(r) = rψ(r), and the equation becomes:
# u''(r) = [2m/ħ² (Ze²/4πε₀r - E) - l(l+1)/r²] u(r)
# where l is the angular momentum quantum number.
# For this problem, we use atomic units where ħ = 1, m = 1, e = 1, and 4πε₀ = 1. This simplifies the equation to:
# u''(r) = [2(1/r) - 2E - l(l+1)/r²] u(r)
# We are computing f(r) = 2(1/r) - 2E - l(l+1)/r² for a given radial grid, energy, and angular momentum quantum number l.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Calculate f(r) for each value in the radial grid
    f_r = (2 / r_grid) - (2 * energy) - (l * (l + 1) / r_grid**2)
    return f_r

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.1', 3)
target = targets[0]

assert np.allclose(f_Schrod(1,1, np.array([0.1,0.2,0.3])), target)
target = targets[1]

assert np.allclose(f_Schrod(2,1, np.linspace(1e-8,100,20)), target)
target = targets[2]

assert np.allclose(f_Schrod(2,3, np.linspace(1e-5,10,10)), target)
