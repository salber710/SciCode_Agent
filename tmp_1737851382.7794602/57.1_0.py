import numpy as np
from scipy import integrate, optimize



# Background: 
# The Schrödinger equation for a quantum harmonic oscillator can be expressed in terms of a dimensionless variable x, 
# where the potential energy term is V(x) = x^2. The energy levels of the harmonic oscillator are quantized and given by 
# E_n = (n + 1/2) * ħω, where n is a non-negative integer. In this problem, the energy E_n is expressed in units of 
# ħω/2, which simplifies the expression to E_n = n + 1/2. The differential equation for the harmonic oscillator can be 
# rewritten as u''(x) = f(x)u(x), where f(x) is a function that depends on the potential V(x) and the energy E_n. 
# Specifically, f(x) = 2(V(x) - E_n) = 2(x^2 - E_n) when the potential is V(x) = x^2. This function f(x) is what we need 
# to compute given x and E_n.

def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    # Calculate f(x) = 2 * (x^2 - En)
    f_x = 2 * (np.square(x) - En)
    return f_x

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('57.1', 3)
target = targets[0]

assert np.allclose(f_x(np.linspace(-5, 5, 10), 1), target)
target = targets[1]

assert np.allclose(f_x(np.linspace(0, 5, 10), 1), target)
target = targets[2]

assert np.allclose(f_x(np.linspace(0, 5, 20), 2), target)
