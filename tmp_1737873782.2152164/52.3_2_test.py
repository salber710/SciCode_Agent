from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''Calculate the derivative of y given r, l and En
    Input 
    y=[u,u'], a list of float where u is the wave function at r, u' is the first derivative of u at r
    r: radius, float
    l: angular momentum quantum number, int
    En: energy, float
    Output
    Schroed: dy/dr=[u',u''] , a 1D array of float where u is the wave function at r, u' is the first derivative of u at r, u'' is the second derivative of u at r
    '''
    # Constants
    Z = 1  # Nuclear charge for hydrogen atom
    a0 = 1  # Bohr radius, set to 1 for simplicity as per instructions

    # Unpack y
    u, up = y  # u is the wave function, up is the first derivative of u

    # Calculate the second derivative of u (u'')
    # Based on the radial part of the Schrödinger equation
    upp = (2 * r * up + (2 * En * r**2 - l * (l + 1)) * u) / r**2

    # Return the derivative of y
    return np.array([up, upp])




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
    # Integrate using solve_ivp from scipy.integrate
    sol = integrate.solve_ivp(
        fun=lambda r, y: Schroed_deriv(y, r, l, En),
        t_span=(R[0], R[-1]),
        y0=y0,
        t_eval=R,
        method='RK45'
    )
    
    # Extract the wave function u from the solution
    u = sol.y[0]
    
    # Normalize the wave function using Simpson's rule
    norm_factor = integrate.simps(u**2, R)
    ur = u / np.sqrt(norm_factor)
    
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
    # Solve the Schrödinger equation to get the wavefunction
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Get the first two points in the radial grid
    r1, r2 = R[0], R[1]
    
    # Get the wavefunction values at those points
    u1, u2 = ur[0], ur[1]
    
    # Divide the wavefunction by r^l
    u1_div = u1 / (r1**l)
    u2_div = u2 / (r2**l)
    
    # Linearly extrapolate to r=0
    # Linear extrapolation formula: f(x) = f(x0) + (x - x0) * (f(x1) - f(x0)) / (x1 - x0)
    # Here we extrapolate for x = 0
    f_at_0 = u1_div + (0 - r1) * (u2_div - u1_div) / (r2 - r1)
    
    return f_at_0


try:
    targets = process_hdf5_to_tuple('52.3', 3)
    target = targets[0]
    assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)

    target = targets[1]
    assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)

    target = targets[2]
    assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e