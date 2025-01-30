from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''Compute the derivatives of the wave function and its first derivative at a given radius, angular momentum, and energy 
    using a dictionary with nested lists.

    Parameters:
    y (dict): A dictionary with a single key 'components' mapping to a list [u, u'], where u is the wave function at r and u' is its first derivative.
    r (float): The radial distance.
    l (int): The angular momentum quantum number.
    En (float): The energy of the system.

    Returns:
    dict: A dictionary with a single key 'derivatives' mapping to a list [u', u''], representing the first and second derivatives.
    '''
    Z = 1  # Atomic number for hydrogen

    # Extract components list from the dictionary
    components = y['components']
    u = components[0]
    u_prime = components[1]

    # Calculate the second derivative of u, u''
    u_double_prime = (l * (l + 1) / r**2 - 2 * Z / r - 2 * En) * u

    # Return the derivatives in a nested list format within a dictionary
    return {'derivatives': [u_prime, u_double_prime]}



# Background: The radial part of the Schrödinger equation for hydrogen can be expressed as a system of linear differential equations. 
# We are dealing with a second-order differential equation that involves the wave function u(r) and its derivative. 
# The task is to integrate this system over a range of r values, using initial conditions. 
# Starting from a large value of r ensures that the boundary conditions are met for bound states. 
# Numerical integration will be performed using Simpson's rule to ensure accuracy in the normalization of the wave function.



def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1d array (linespace) of radius (float)
    Output
    ur: the integration result, float
    '''

    # Function to compute the derivatives at each step
    def derivs(y, r):
        dy = Schroed_deriv({'components': y}, r, l, En)
        return dy['derivatives']

    # Integrate using solve_ivp with 'RK45' method
    sol = integrate.solve_ivp(derivs, [R[0], R[-1]], y0, t_eval=R, method='RK45')

    # Extract the wave function values (first component of y) from the solution
    u_values = sol.y[0]

    # Normalize the wave function using Simpson's rule
    norm_factor = integrate.simpson(u_values**2, R)
    ur = u_values / np.sqrt(norm_factor)

    return ur


try:
    targets = process_hdf5_to_tuple('52.2', 3)
    target = targets[0]
    y0 = [0, -1e-5]
    En = 1.0
    l = 1
    R = np.logspace(-2,3.2,100)
    assert np.allclose(SolveSchroedinger(y0,En,l,R), target)

    target = targets[1]
    y0 = [0, -1e-5]
    En = 1.5
    l = 2
    R = np.logspace(-1,3.2,100)
    assert np.allclose(SolveSchroedinger(y0,En,l,R), target)

    target = targets[2]
    y0 = [0, -1e-5]
    En = 2.5
    l = 2
    R = np.logspace(1,3.2,100)
    assert np.allclose(SolveSchroedinger(y0,En,l,R), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e