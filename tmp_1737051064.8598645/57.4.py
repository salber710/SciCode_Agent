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


# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form 
# u''(x) = f(x)u(x). It is particularly useful for solving the Schrödinger equation in quantum mechanics, where the potential 
# function f(x) is known, and we seek the wave function u(x). The method is based on a finite difference approach that provides 
# a stable and accurate solution by considering the values of the function and its derivatives at discrete points. The Numerov 
# method uses a predictor-corrector scheme to iteratively compute the values of u(x) over a grid of x values, given initial 
# boundary conditions for u and its derivative.

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
    
    # Numerov's method iteration
    for i in range(1, n_points - 1):
        u[i + 1] = (2 * (1 - (5/12) * step**2 * f_in[i]) * u[i] - 
                    (1 + (1/12) * step**2 * f_in[i-1]) * u[i-1]) / (1 + (1/12) * step**2 * f_in[i+1])
    
    return u


# Background: In quantum mechanics, solving the Schrödinger equation for a harmonic oscillator involves finding the wave function u(x) 
# that satisfies the equation u''(x) = f(x)u(x), where f(x) = E_n - x^2. The Numerov method is a numerical technique used to solve 
# such second-order differential equations. Once the wave function u(x) is obtained, it is important to normalize it so that the 
# total probability of finding the particle is 1. This is done using numerical integration, such as Simpson's rule, which is 
# implemented in the scipy.integrate.simpson function. The task is to wrap the previously defined functions for calculating f(x) 
# and solving the differential equation using the Numerov method into a single function that also normalizes the resulting wave 
# function.



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
    
    # Calculate f(x) = En - x^2
    f_in = En - np.square(x)
    
    # Solve the differential equation using the Numerov method
    n_points = len(x)
    u = np.zeros(n_points)
    u[0] = u_b
    u[1] = u_b + step * up_b + (step**2 / 2) * f_in[0] * u_b
    
    for i in range(1, n_points - 1):
        u[i + 1] = (2 * (1 - (5/12) * step**2 * f_in[i]) * u[i] - 
                    (1 + (1/12) * step**2 * f_in[i-1]) * u[i-1]) / (1 + (1/12) * step**2 * f_in[i+1])
    
    # Normalize the wave function using Simpson's rule
    norm_factor = integrate.simpson(u**2, x)
    u_norm = u / np.sqrt(norm_factor)
    
    return u_norm



# Background: In quantum mechanics, the number of nodes (points where the wave function crosses zero) in the wave function 
# of a quantum harmonic oscillator is related to the energy level of the system. Specifically, the number of nodes is equal 
# to the quantum number n. Counting the number of sign changes in the wave function is a way to determine the number of nodes, 
# as each sign change indicates a crossing through zero. This is useful for verifying the correctness of the computed wave 
# function and ensuring it corresponds to the expected energy level.

def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    # Calculate the differences in sign between consecutive elements
    sign_changes = np.sum(np.diff(np.sign(solv_schrod)) != 0)
    
    return sign_changes


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('57.4', 3)
target = targets[0]

assert np.allclose(count_sign_changes(np.array([-1,2,-3,4,-5])), target)
target = targets[1]

assert np.allclose(count_sign_changes(np.array([-1,-2,-3,-4,-5])), target)
target = targets[2]

assert np.allclose(count_sign_changes(np.array([0,-2,3,-4,-5])), target)
