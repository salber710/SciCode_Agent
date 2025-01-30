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




def Solve_Schrod(x, En, u_b, up_b, step):
    # Define the potential function for a harmonic oscillator
    V = 0.5 * x**2
    
    # Define f(x) for the Numerov method
    f_x = 2 * (V - En)
    
    # Initialize the solution array
    u = np.zeros_like(x)
    u[0] = u_b
    u[1] = u_b + step * up_b
    
    # Numerov method constants
    h2 = step**2
    h12 = h2 / 12.0
    
    # Numerov integration using a loop with precomputed coefficients
    coeff1 = 1 - 5 * h12 * f_x
    coeff2 = 1 + h12 * np.roll(f_x, -1)
    coeff3 = 1 + h12 * np.roll(f_x, 1)
    
    for i in range(1, len(x) - 1):
        u[i + 1] = (2 * u[i] * coeff1[i] - u[i - 1] * coeff3[i]) / coeff2[i]
    
    # Normalize the wave function using Simpson's rule
    normalization_factor = np.sqrt(simpson(u**2, x))
    u_norm = u / normalization_factor
    
    return u_norm


def count_sign_changes(solv_schrod):
    return sum((x < 0) ^ (y < 0) for x, y in zip(solv_schrod, solv_schrod[1:]))



# Background: The Shooting method is a numerical technique used to find the eigenvalues (bound state energies) of differential equations, such as the Schrödinger equation for quantum systems. In this context, we are interested in finding the bound states of a quantum harmonic oscillator. The method involves guessing an energy value, solving the Schrödinger equation using the Numerov method, and checking the boundary conditions. If the solution changes sign the correct number of times (related to the quantum number n), it indicates a bound state. We start from zero energy and incrementally increase the energy by a specified step size until the maximum energy is reached. The principal quantum number n corresponds to the number of nodes (sign changes) in the wave function.

def BoundStates(x, Emax, Estep):
    '''Input
    x: coordinate x; a float or a 1D array of float
    Emax: maximum energy of a bound state; a float
    Estep: energy step size; a float
    Output
    bound_states: a list, each element is a tuple containing the principal quantum number (an int) and energy (a float)
    '''



    # Initialize the list to store bound states
    bound_states = []

    # Initial conditions for the wave function
    u_b = 0.0  # u(x=0) = 0
    up_b = 1.0  # u'(x=0) = 1 (arbitrary non-zero value)

    # Iterate over energy values from 0 to Emax with step Estep
    En = 0.0
    while En <= Emax:
        # Solve the Schrödinger equation using the Solve_Schrod function
        u_norm = Solve_Schrod(x, En, u_b, up_b, Estep)

        # Count the number of sign changes in the wave function
        n_nodes = count_sign_changes(u_norm)

        # Check if the number of nodes corresponds to a new bound state
        if len(bound_states) == 0 or bound_states[-1][0] != n_nodes:
            bound_states.append((n_nodes, En))

        # Increment the energy
        En += Estep

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