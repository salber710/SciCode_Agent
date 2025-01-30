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
    def derivs(r, y):
        dy = Schroed_deriv({'components': y}, r, l, En)
        return np.array(dy['derivatives'])

    # Initialize the wave function and its derivative
    y = np.array(y0)
    
    # Prepare an array to store the integration results
    u_values = np.zeros(len(R))
    
    # Perform the integration using the midpoint method
    for i in range(1, len(R)):
        dr = R[i] - R[i-1]
        mid_r = R[i-1] + dr/2
        mid_y = y + (dr/2) * derivs(R[i-1], y)
        y += dr * derivs(mid_r, mid_y)
        u_values[i] = y[0]  # Store the wave function value

    # Normalize the wave function using the rectangle (midpoint) rule
    integral = np.sum(u_values**2 * np.diff(R, prepend=R[0]))
    ur = u_values / np.sqrt(integral)

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