from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize

def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''

    # In the context of the quantum harmonic oscillator problem, the function f(x) in the equation u''(x) = f(x)u(x)
    # is given by f(x) = E_n - V(x), where V(x) is the potential energy function V(x) = x^2.
    # Therefore, f(x) = E_n - x^2.

    # Calculate f(x)
    f_x = En - x**2

    return f_x




def Numerov(f_in, u_b, up_b, step):
    '''Given precomputed function f(x), solve the differential equation u''(x) = f(x)*u(x)
    using the Numerov method.
    Inputs:
    - f_in: input function f(x); a 1D array of float representing the function values at discretized points
    - u_b: the value of u at boundary; a float
    - up_b: the derivative of u at boundary; a float
    - step: step size; a float.
    Output:
    - u: u(x); a 1D array of float representing the solution.
    '''

    # Number of points
    N = len(f_in)
    
    # Initialize the array for u with zeros
    u = np.zeros(N)
    
    # Initial conditions
    u[0] = u_b
    u[1] = u_b + up_b * step
    
    # Coefficient used in the Numerov method
    k = step**2 / 12.0
    
    # Apply the Numerov method
    for i in range(1, N-1):
        u[i+1] = (2 * (1 - 5 * k * f_in[i]) * u[i] - (1 + k * f_in[i-1]) * u[i-1]) / (1 + k * f_in[i+1])
    
    return u


try:
    targets = process_hdf5_to_tuple('57.2', 3)
    target = targets[0]
    assert np.allclose(Numerov(f_x(np.linspace(0,5,10), 1.0), 1.0, 0.0, np.linspace(0,5,10)[0]-np.linspace(0,5,10)[1]), target)

    target = targets[1]
    assert np.allclose(Numerov(f_x(np.linspace(0,5,100), 1.0), 1.0, 0.0, np.linspace(0,5,100)[0]-np.linspace(0,5,100)[1]), target)

    target = targets[2]
    assert np.allclose(Numerov(f_x(np.linspace(0,5,100), 3.0), 0.0, 1.0, np.linspace(0,5,100)[0]-np.linspace(0,5,100)[1]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e