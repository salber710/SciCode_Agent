import numpy as np
from scipy import integrate, optimize



# Background: In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes over time. 
# For a harmonic oscillator, the potential energy is given by V(x) = (1/2) m ω^2 x^2. By scaling the variable x and the energy E_n, 
# we can simplify the equation. In this problem, the potential term is scaled to V(x) = x^2, and the energy E_n is expressed in units 
# of (ħω/2). The Schrödinger equation for the harmonic oscillator can be rewritten in terms of a dimensionless form where the 
# second derivative of the wave function u(x) is related to a function f(x) by u''(x) = f(x)u(x). The function f(x) is given by 
# f(x) = E_n - x^2, where E_n is the scaled energy level of the harmonic oscillator. This function f(x) represents the effective 
# potential in the dimensionless form of the Schrödinger equation.

def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    # Calculate f(x) = En - x^2
    f_x = En - np.square(x)
    return f_x


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('57.1', 3)
target = targets[0]

assert np.allclose(f_x(np.linspace(-5, 5, 10), 1), target)
target = targets[1]

assert np.allclose(f_x(np.linspace(0, 5, 10), 1), target)
target = targets[2]

assert np.allclose(f_x(np.linspace(0, 5, 20), 2), target)
