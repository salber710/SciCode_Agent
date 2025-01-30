from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u, u_prime = y
    u_double_prime = u * (l * (l + 1) / r**2 - 2 / r + 2 * En)
    return [u_prime, u_double_prime]



# Background: 
# The Schrödinger equation in quantum mechanics describes how the quantum state of a physical system changes over time. 
# For a spherically symmetric potential, the radial part of the Schrödinger equation can be expressed in terms of a 
# second-order differential equation. In this context, we are dealing with the hydrogen atom where the potential is 
# Coulombic, and we have already expressed the radial part of the Schrödinger equation as a system of first-order 
# differential equations. The task now is to integrate these equations over a range of radial distances (r) starting 
# from a large value of r, which is typically done to ensure the wave function behaves correctly at infinity. 
# After performing the integration, the result needs to be normalized. Simpson's rule is a method for numerical 
# integration that provides an approximation of the integral of a function. It is particularly useful for integrating 
# functions that are smooth over the interval of integration.



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
    
    # Define the function to compute the derivatives using the provided Schroed_deriv function
    def derivs(r, y):
        return Schroed_deriv(y, r, l, En)
    
    # Perform the integration using solve_ivp, starting from large r
    sol = integrate.solve_ivp(derivs, [R[-1], R[0]], y0, t_eval=R[::-1], method='RK45')
    
    # Extract the solution for u(r)
    u_r = sol.y[0][::-1]  # Reverse the solution to match the original order of R
    
    # Normalize the result using Simpson's rule
    norm_factor = integrate.simps(u_r**2, R)
    ur = u_r / np.sqrt(norm_factor)
    
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