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
    
    # Initialize the solution array
    u = np.zeros(N)
    
    # Initial conditions
    u[0] = u_b
    u[1] = u_b + step * up_b
    
    # Numerov's method coefficients
    step_sq = step**2
    factor = step_sq / 12.0
    
    # Iterate using Numerov's method
    for i in range(1, N - 1):
        u[i + 1] = ((2 * (1 - 5 * factor * f_in[i]) * u[i]) - 
                    (1 + factor * f_in[i - 1]) * u[i - 1]) / (1 + factor * f_in[i + 1])
    
    return u





def f_x(x, En):
    '''Return the value of f(x) with energy En
    Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    Output
    f_x: the value of f(x); a float or a 1D array of float
    '''
    # Calculate f(x)
    return En - x**2

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
    
    # Initial conditions
    u[0] = u_b
    u[1] = u_b + step * up_b
    
    # Numerov's method coefficients
    step_sq = step**2
    factor = step_sq / 12.0
    
    # Iterate using Numerov's method
    for i in range(1, N - 1):
        u[i + 1] = ((2 * (1 - 5 * factor * f_in[i]) * u[i]) - 
                    (1 + factor * f_in[i - 1]) * u[i - 1]) / (1 + factor * f_in[i + 1])
    
    return u

def Solve_Schrod(x, En, u_b, up_b, step):
    '''Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    u_b: value of u(x) at one boundary for the Numverov function; a float
    up_b: value of the derivative of u(x) at one boundary for the Numverov function; a float
    step: the step size for the Numerov method; a float
    Output
    u_norm: normalized u(x); a float or a 1D array of float
    '''
    # Compute f(x) for given x and energy En
    f_values = f_x(x, En)
    
    # Use Numerov method to solve for u(x)
    u_values = Numerov(f_values, u_b, up_b, step)
    
    # Normalize u(x) using Simpson's rule
    norm_const = np.sqrt(integrate.simpson(u_values**2, x))
    u_norm = u_values / norm_const
    
    return u_norm


try:
    targets = process_hdf5_to_tuple('57.3', 3)
    target = targets[0]
    x = np.linspace(0,5,20)
    assert np.allclose(Solve_Schrod(x, 1.0, 1.0, 0.0, x[0]-x[1]), target)

    target = targets[1]
    x = np.linspace(0,5,100)
    assert np.allclose(Solve_Schrod(x, 7.0, 0.0, 1.0, x[0]-x[1]), target)

    target = targets[2]
    x = np.linspace(0,5,100)
    assert np.allclose(Solve_Schrod(x, 5.0, 1.0, 0.0, x[0]-x[1]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e