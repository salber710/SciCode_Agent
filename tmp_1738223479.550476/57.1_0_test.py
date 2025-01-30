from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize



# Background: The Schrödinger equation for the quantum harmonic oscillator can be rewritten in a dimensionless form.
# In this form, the potential term is given by V(x) = x^2, where x is the scaled position. The energy levels 
# of the harmonic oscillator are quantized and given by E_n = (n + 1/2), where n is a non-negative integer, 
# and E_n is expressed in units of (ħω/2). The differential equation for the oscillator can be rewritten as 
# u''(x) = f(x)u(x), where f(x) = 2(V(x) - E_n). In this dimensionless form, the task is to determine the 
# function f(x) based on the input position x and energy E_n. This requires evaluating the expression 
# f(x) = 2(x^2 - E_n), where x^2 represents the potential energy term and E_n is the energy term.

def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    
    # Calculate f(x) as 2 * (V(x) - En), where V(x) = x^2
    f_x = 2 * (np.square(x) - En)
    
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