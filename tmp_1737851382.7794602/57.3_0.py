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


# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form u''(x) = f(x)u(x). 
# It is particularly useful for problems in quantum mechanics, such as the Schrödinger equation, where the potential and energy terms can be 
# combined into a function f(x). The method is based on a finite difference approach that provides a stable and accurate solution by considering 
# the values of the function and its second derivative at discrete points. The Numerov method uses a three-point recursion relation to compute 
# the solution at each step, given the initial conditions and the precomputed values of f(x).


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
    if step <= 0:
        raise ValueError("Step size must be positive.")

    # Number of points
    N = len(f_in)
    
    # Initialize the solution array
    u = np.zeros(N)
    
    # Set initial conditions
    u[0] = u_b
    u[1] = u_b + step * up_b
    
    # Precompute step squared
    step2 = step**2
    
    # Numerov's method iteration
    for i in range(1, N-1):
        u[i+1] = (2 * (1 - 5 * step2 * f_in[i] / 12) * u[i] - (1 + step2 * f_in[i-1] / 12) * u[i-1]) / (1 + step2 * f_in[i+1] / 12)
    
    return u



# Background: The Schrödinger equation for a quantum harmonic oscillator can be solved numerically using the Numerov method. 
# This involves computing the function f(x) = 2(x^2 - E_n) and using it to solve the differential equation u''(x) = f(x)u(x). 
# The solution u(x) obtained from the Numerov method needs to be normalized to ensure that the total probability is 1. 
# This normalization can be achieved using Simpson's rule, which is a numerical integration technique. 
# The scipy.integrate.simpson function can be used to perform this integration over the computed u(x) values.



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
    
    # Calculate f(x) using the provided f_x function logic
    f_in = 2 * (np.square(x) - En)
    
    # Solve for u(x) using the Numerov method
    N = len(f_in)
    u = np.zeros(N)
    u[0] = u_b
    u[1] = u_b + step * up_b
    step2 = step**2
    
    for i in range(1, N-1):
        u[i+1] = (2 * (1 - 5 * step2 * f_in[i] / 12) * u[i] - (1 + step2 * f_in[i-1] / 12) * u[i-1]) / (1 + step2 * f_in[i+1] / 12)
    
    # Normalize u(x) using Simpson's rule
    integral = integrate.simpson(np.square(u), x)
    u_norm = u / np.sqrt(integral)
    
    return u_norm

from scicode.parse.parse import process_hdf5_to_tuple
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
