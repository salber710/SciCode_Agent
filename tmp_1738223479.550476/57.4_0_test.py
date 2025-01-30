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

    # Set initial conditions using an alternative Taylor expansion
    u[0] = u_b
    u[1] = u_b + step * up_b - (step**2 / 4) * f_in[0] * u_b

    # Precompute factors for the Numerov method
    step_squared = step * step
    coeff = step_squared / 12.0

    # Use a unique approach with modified coefficients
    for i in range(1, N-1):
        a = 1 + coeff * f_in[i-1]
        b = 2 - 5 * coeff * f_in[i]
        c = 1 - coeff * f_in[i+1]

        # Update using a different arrangement of the Numerov formula
        u[i+1] = (b * u[i] - a * u[i-1]) / c

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

    # Harmonic oscillator potential function
    def harmonic_potential(x):
        return np.power(x, 2)

    # Calculate f(x) using potential and energy
    def f_x(x, En):
        return 2 * (harmonic_potential(x) - En)

    # Implementation of Numerov's method
    def numerov(f_values, u_init, du_init, step_size):
        n_points = len(f_values)
        u_values = np.zeros(n_points)
        u_values[0] = u_init
        u_values[1] = u_init + step_size * du_init - (step_size**2 / 6) * f_values[0] * u_init

        step_sq = step_size**2 / 12.0

        for i in range(1, n_points - 1):
            u_values[i + 1] = ((2 * (1 - 5 * step_sq * f_values[i]) * u_values[i]) -
                               ((1 + step_sq * f_values[i - 1]) * u_values[i - 1])) / \
                              (1 + step_sq * f_values[i + 1])

        return u_values

    # Compute f(x) for the given x and energy
    f_values = f_x(x, En)

    # Solve the differential equation using Numerov's method
    u_values = numerov(f_values, u_b, up_b, step)

    # Normalize the solution using Simpson's rule
    norm_factor = simpson(u_values**2, x)
    u_normalized = u_values / np.sqrt(norm_factor)

    return u_normalized



# Background: In numerical analysis and signal processing, a sign change between consecutive elements
# in an array indicates that the values have crossed zero. This can be especially useful in physics,
# such as in quantum mechanics, where the number of sign changes in a wave function is related to 
# the number of nodes, which in turn is associated with the quantum number of the state. Counting 
# these sign changes can provide insight into the properties of the solution obtained from the 
# Schrodinger equation.

def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    # Initialize the count of sign changes
    sign_changes = 0

    # Iterate over the array and count sign changes between consecutive elements
    for i in range(1, len(solv_schrod)):
        # Check if the product of consecutive elements is negative, indicating a sign change
        if solv_schrod[i] * solv_schrod[i-1] < 0:
            sign_changes += 1

    return sign_changes


try:
    targets = process_hdf5_to_tuple('57.4', 3)
    target = targets[0]
    assert np.allclose(count_sign_changes(np.array([-1,2,-3,4,-5])), target)

    target = targets[1]
    assert np.allclose(count_sign_changes(np.array([-1,-2,-3,-4,-5])), target)

    target = targets[2]
    assert np.allclose(count_sign_changes(np.array([0,-2,3,-4,-5])), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e