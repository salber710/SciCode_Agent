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



def Numerov(f_in, u_b, up_b, step):
    n = len(f_in)
    u = np.zeros(n)
    u[0] = u_b
    u[1] = u_b + step * up_b + 0.5 * step**2 * f_in[0] * u_b

    h2 = step**2
    h12 = h2 / 12.0

    # Compute the coefficients for the Numerov formula
    coeff = np.zeros((n, 3))  # coeff[i] = [g[i-1], f[i], g[i+1]]
    coeff[:, 1] = 1 - 5 * h12 * f_in  # f[i]
    coeff[:-1, 2] = 1 + h12 * f_in[1:]  # g[i+1]
    coeff[1:, 0] = 1 + h12 * f_in[:-1]  # g[i-1]

    for i in range(1, n-1):
        u[i+1] = (2 * u[i] * coeff[i, 1] - u[i-1] * coeff[i, 0]) / coeff[i, 2]

    return u



# Background: The Schrödinger equation for a quantum harmonic oscillator can be solved numerically using the Numerov method. 
# This method is particularly useful for solving second-order differential equations of the form u''(x) = f(x)u(x). 
# The function f(x) is derived from the potential energy and the energy level of the system. 
# Once the wave function u(x) is obtained, it must be normalized to ensure that the total probability of finding the particle is 1. 
# Normalization is achieved using Simpson's rule, which is a numerical integration technique that provides an accurate estimate of the integral of a function.



def Solve_Schrod(x, En, u_b, up_b, step):
    '''Input
    x: coordinate x; a float or a 1D array of float
    En: energy; a float
    u_b: value of u(x) at one boundary for the Numerov function; a float
    up_b: value of the derivative of u(x) at one boundary for the Numerov function; a float
    step: the step size for the Numerov method; a float
    Output
    u_norm: normalized u(x); a float or a 1D array of float
    '''
    
    # Calculate f(x) using the provided energy En
    f_in = En + (-x)**2
    
    # Use the Numerov method to solve for u(x)
    n = len(f_in)
    u = np.zeros(n)
    u[0] = u_b
    u[1] = u_b + step * up_b + 0.5 * step**2 * f_in[0] * u_b

    h2 = step**2
    h12 = h2 / 12.0

    # Compute the coefficients for the Numerov formula
    coeff = np.zeros((n, 3))  # coeff[i] = [g[i-1], f[i], g[i+1]]
    coeff[:, 1] = 1 - 5 * h12 * f_in  # f[i]
    coeff[:-1, 2] = 1 + h12 * f_in[1:]  # g[i+1]
    coeff[1:, 0] = 1 + h12 * f_in[:-1]  # g[i-1]

    for i in range(1, n-1):
        u[i+1] = (2 * u[i] * coeff[i, 1] - u[i-1] * coeff[i, 0]) / coeff[i, 2]

    # Normalize the wave function using Simpson's rule
    integral = integrate.simpson(u**2, x)
    u_norm = u / np.sqrt(integral)

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