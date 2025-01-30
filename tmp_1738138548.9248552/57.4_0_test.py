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



# Background: In quantum mechanics, the number of sign changes in the wave function of a quantum harmonic oscillator is related to the number of nodes, which corresponds to the quantum number of the state. A node is a point where the wave function passes through zero, indicating a change in sign. Counting the number of sign changes in a wave function can help identify the energy level or quantum state of the system. This is important for understanding the properties of the wave function and the corresponding energy levels.

def count_sign_changes(solv_schrod):
    '''Input
    solv_schrod: a 1D array
    Output
    sign_changes: number of times of sign change occurrence; an int
    '''
    # Calculate the differences in sign between consecutive elements
    sign_changes = np.sum(np.diff(np.sign(solv_schrod)) != 0)
    
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