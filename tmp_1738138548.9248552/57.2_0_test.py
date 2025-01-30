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
    return En + (-x)**2



# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form u''(x) = f(x)u(x). 
# It is particularly useful for solving the time-independent Schrödinger equation in quantum mechanics. The method is based on a finite difference 
# approach that provides a stable and accurate solution by considering the function values at three consecutive points. The key idea is to 
# approximate the second derivative using a Taylor expansion and to incorporate the function f(x) into the update formula. The Numerov method 
# requires initial conditions, which are typically the value of the function and its derivative at a boundary. The method iteratively computes 
# the solution across a discretized domain using a specified step size.

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
    n_points = len(f_in)
    
    # Initialize the solution array
    u = np.zeros(n_points)
    
    # Set initial conditions
    u[0] = u_b
    u[1] = u_b + step * up_b + (step**2 / 2) * f_in[0] * u_b
    
    # Coefficient for the Numerov method
    coeff = step**2 / 12
    
    # Iterate over the range to compute the solution using the Numerov method
    for i in range(1, n_points - 1):
        u[i + 1] = (2 * (1 - 5 * coeff * f_in[i]) * u[i] - (1 + coeff * f_in[i - 1]) * u[i - 1]) / (1 + coeff * f_in[i + 1])
    
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