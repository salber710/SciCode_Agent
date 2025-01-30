from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize



# Background: In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes over time. 
# For a harmonic oscillator, the potential energy is given by V(x) = (1/2) m ω^2 x^2. By scaling the variable x and the energy E_n, 
# we can simplify the potential to V(x) = x^2 and express the energy in units of (ħω/2). The time-independent Schrödinger equation 
# for a harmonic oscillator can be rewritten in terms of a dimensionless form. In this form, the equation becomes u''(x) = f(x)u(x), 
# where f(x) = E_n - x^2. This function f(x) represents the effective potential energy term in the differential equation.

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


try:
    targets = process_hdf5_to_tuple('57.1', 3)
    target = targets[0]
    assert np.allclose(f_x(np.linspace(-5, 5, 10), 1), target)

    target = targets[1]
    assert np.allclose(f_x(np.linspace(0, 5, 10), 1), target)

    target = targets[2]
    assert np.allclose(f_x(np.linspace(0, 5, 20), 2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e