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
    u_values[0] = y0[0]

    # Perform the integration using the Verlet method
    y_next = y + (R[1] - R[0]) * derivs(R[0], y)  # Initial step
    for i in range(1, len(R)):
        dr = R[i] - R[i-1]
        y_new = 2 * y_next - y + dr**2 * derivs(R[i], y_next)
        u_values[i] = y_new[0]
        y, y_next = y_next, y_new

    # Normalize the wave function using the Boole's rule
    if len(R) < 5 or (len(R) - 1) % 4 != 0:
        raise ValueError("Boole's rule requires (n-1) to be a multiple of 4")
    
    n = len(R)
    h = (R[-1] - R[0]) / (n - 1)
    integral = (2 * h / 45) * (
        7 * u_values[0]**2 +
        32 * np.sum(u_values[1:n-1:4]**2) +
        12 * np.sum(u_values[2:n-1:4]**2) +
        32 * np.sum(u_values[3:n-1:4]**2) +
        14 * np.sum(u_values[4:n-1:4]**2) +
        7 * u_values[-1]**2
    )
    ur = u_values / np.sqrt(integral)

    return ur


def Shoot(En, R, l, y0):
    '''Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    Output 
    f_at_0: Extrapolate u(0), float
    '''
    # Calculate the wavefunction values using SolveSchroedinger
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Extract the first two wavefunction values and their radii
    u_values = ur[:2]
    r_values = R[:2]
    
    # Adjust the wavefunction values using a custom function
    def adjust_wavefunction(u, r, l):
        return u / (r ** l)

    adjusted_values = list(map(adjust_wavefunction, u_values, r_values, [l, l]))

    # Perform the linear extrapolation using a reversed approach
    # Reverse the order of subtraction and use a negative slope for uniqueness
    slope = -(adjusted_values[1] - adjusted_values[0]) / (r_values[1] - r_values[0])
    f_at_0 = adjusted_values[1] + slope * (r_values[1])

    return f_at_0





def FindBoundStates(y0, R, l, nmax, Esearch):
    '''Input
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    nmax: maximum number of bounds states wanted, int
    Esearch: energy mesh used for search, an 1D array of float
    Output
    Ebnd: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    Ebnd = []

    def boundary_condition(En):
        return Shoot(En, R, l, y0)

    # Loop over the energy search mesh
    for i in range(len(Esearch) - 1):
        E_start, E_end = Esearch[i], Esearch[i + 1]
        
        # Use root_scalar with the 'ridder' method to find roots
        sol = root_scalar(boundary_condition, bracket=[E_start, E_end], method='ridder')
        
        if sol.converged:
            Ebnd.append((l, sol.root))
            if len(Ebnd) >= nmax:
                break

    return Ebnd


try:
    targets = process_hdf5_to_tuple('52.4', 3)
    target = targets[0]
    y0 = [0, -1e-5]
    Esearch = -1.2/np.arange(1,20,0.2)**2
    R = np.logspace(-6,2.2,500)
    nmax=7
    Bnd=[]
    for l in range(nmax-1):
        Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
    assert np.allclose(Bnd, target)

    target = targets[1]
    Esearch = -0.9/np.arange(1,20,0.2)**2
    R = np.logspace(-8,2.2,1000)
    nmax=5
    Bnd=[]
    for l in range(nmax-1):
        Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
    assert np.allclose(Bnd, target)

    target = targets[2]
    Esearch = -1.2/np.arange(1,20,0.2)**2
    R = np.logspace(-6,2.2,500)
    nmax=7
    Bnd=[]
    for l in range(nmax-1):
        Bnd += FindBoundStates(y0,R,l,nmax-l,Esearch)
    assert np.isclose(Bnd[0], (0,-1)).any() == target.any()

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e