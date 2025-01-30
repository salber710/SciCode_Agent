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
    
    # Define a function using a functional programming approach with reduce

    
    def calculate_fx(x_val, energy):
        # Calculate potential energy V(x) = x^2
        V_x = reduce(lambda a, b: a * b, [x_val, x_val])
        return 2 * (V_x - energy)
    
    # Check if x is iterable by checking for the '__iter__' attribute
    if hasattr(x, '__iter__'):
        return [calculate_fx(xi, En) for xi in x]
    else:
        return calculate_fx(x, En)



# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form u''(x) = f(x)u(x). 
# This method is particularly useful for solving the time-independent Schrödinger equation in quantum mechanics. 
# It is a predictor-corrector method that provides higher accuracy by incorporating second-derivative information into the finite difference scheme. 
# The key idea is to express u(x) using a central difference approximation which involves the values of the function at multiple points.
# Specifically, the Numerov method updates the solution using the formula:
# u[i+1] = (2(1 - 5/12 * step^2 * f[i]) * u[i] - (1 + 1/12 * step^2 * f[i-1]) * u[i-1]) / (1 + 1/12 * step^2 * f[i+1])
# This requires initial conditions, such as the value of u and its derivative at a boundary point, to propagate the solution over a grid.


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
    
    # Initialize the solution array
    u = np.zeros(N)
    
    # Set initial conditions
    u[0] = u_b
    u[1] = u_b + step * up_b

    # Coefficient for the Numerov method
    coeff = step**2 / 12.0
    
    # Iterate over the range 1 to N-2 (as we need u[i+1], u[i], and u[i-1])
    for i in range(1, N-1):
        # Numerov's update formula
        u[i+1] = (
            (2 * (1 - coeff * f_in[i]) * u[i] - (1 + coeff * f_in[i-1]) * u[i-1])
            / (1 + coeff * f_in[i+1])
        )

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