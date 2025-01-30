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
    N = len(f_in)
    u = np.zeros(N)
    u[0] = u_b
    u[1] = u_b + step * up_b
    step_sq = step**2
    factor = step_sq / 12.0

    for i in range(1, N - 1):
        u[i + 1] = ((2 * (1 - 5 * factor * f_in[i]) * u[i]) - 
                    (1 + factor * f_in[i - 1]) * u[i - 1]) / (1 + factor * f_in[i + 1])
    
    return u

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
    # Calculate f(x) for the given x and En
    f_values = f_x(x, En)
    
    # Use the Numerov method to solve for u(x)
    u = Numerov(f_values, u_b, up_b, step)
    
    # Normalize the result using Simpson's rule
    norm_factor = integrate.simpson(u**2, x)
    u_norm = u / np.sqrt(norm_factor)
    
    return u_norm


def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    # Initialize the counter for sign changes
    sign_changes = 0
    
    # Iterate through the array, comparing each element with the next one
    for i in range(len(solv_schrod) - 1):
        # If the product of the current and next element is negative, a sign change occurred
        if solv_schrod[i] * solv_schrod[i + 1] < 0:
            sign_changes += 1
    
    return sign_changes





def BoundStates(x, Emax, Estep):
    '''Input
    x: coordinate x; a float or a 1D array of float
    Emax: maximum energy of a bound state; a float
    Estep: energy step size; a float
    Output
    bound_states: a list, each element is a tuple containing the principal quantum number (an int) and energy (a float)
    '''

    def f_x(x, En):
        '''Return the value of f(x) with energy En'''
        return En - x**2

    def Numerov(f_in, u_b, up_b, step):
        '''Solve the differential equation u''(x) = f(x)*u(x) using the Numerov method.'''
        N = len(f_in)
        u = np.zeros(N)
        u[0] = u_b
        u[1] = u_b + step * up_b
        step_sq = step**2
        factor = step_sq / 12.0

        for i in range(1, N - 1):
            u[i + 1] = ((2 * (1 - 5 * factor * f_in[i]) * u[i]) - 
                        (1 + factor * f_in[i - 1]) * u[i - 1]) / (1 + factor * f_in[i + 1])
        
        return u

    def Solve_Schrod(x, En, u_b, up_b, step):
        '''Solve the Schrodinger equation and normalize the result.'''
        f_values = f_x(x, En)
        u = Numerov(f_values, u_b, up_b, step)
        norm_factor = integrate.simpson(u**2, x)
        u_norm = u / np.sqrt(norm_factor)
        return u_norm

    def count_sign_changes(solv_schrod):
        '''Count the number of sign changes in a 1D array.'''
        sign_changes = 0
        for i in range(len(solv_schrod) - 1):
            if solv_schrod[i] * solv_schrod[i + 1] < 0:
                sign_changes += 1
        return sign_changes

    bound_states = []
    n = 0
    for En in np.arange(0, Emax, Estep):
        u_b = 0.0  # Boundary condition at x[0]
        up_b = 1.0  # Initial slope at x[0]
        u_norm = Solve_Schrod(x, En, u_b, up_b, x[1] - x[0])
        sign_changes = count_sign_changes(u_norm)
        
        if sign_changes == n:
            bound_states.append((n, En))
            n += 1

    return bound_states


try:
    targets = process_hdf5_to_tuple('57.5', 3)
    target = targets[0]
    assert np.allclose(BoundStates(np.linspace(0,10,200), 2, 1e-4), target)

    target = targets[1]
    assert np.allclose(BoundStates(np.linspace(0,5,100), 1, 1e-4), target)

    target = targets[2]
    assert np.allclose(BoundStates(np.linspace(0,20,400), 11.1, 1e-4), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e