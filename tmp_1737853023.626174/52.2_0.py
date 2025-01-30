import numpy as np
from scipy import integrate, optimize

# Background: 
# The radial part of the Schrödinger equation for a hydrogen-like atom can be expressed in terms of a second-order differential equation. 
# For a given angular momentum quantum number `l` and energy `En`, the equation is:
# 
# d²u/dr² = [l(l+1)/r² - 2/r + 2*En] * u
# 
# Here, `u(r)` is the radial wave function, and `u'(r)` is its first derivative. 
# We can express this second-order differential equation as a system of first-order differential equations:
# 
# Let y = [u, u'], then:
# dy/dr = [u', u'']
# where u'' = [l(l+1)/r² - 2/r + 2*En] * u
# 
# This transformation allows us to use numerical methods to solve the system of equations.


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
    u, u_prime = y
    if r <= 0:
        raise ValueError("Radius r must be greater than 0 to avoid division by zero.")
    # Calculate the second derivative of u using the radial Schrödinger equation
    u_double_prime = (l * (l + 1) / r**2 - 2 / r + 2 * En) * u
    # Return the derivatives as a numpy array
    Schroed = np.array([u_prime, u_double_prime])
    return Schroed



# Background: 
# To solve the radial part of the Schrödinger equation numerically, we need to integrate the system of first-order differential equations derived from the second-order differential equation. 
# The integration will be performed over a range of radii, starting from a large value of r and moving inward. 
# This approach is often used because the asymptotic behavior of the wave function at large r is known and can be used as a boundary condition.
# After obtaining the solution over the range, the wave function needs to be normalized. 
# Simpson's rule is a numerical method for approximating the integral of a function, which can be used to normalize the wave function.

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
    # Define the function to compute the derivatives using Schroed_deriv
    def dydr(r, y):
        return Schroed_deriv(y, r, l, En)
    
    # Perform the integration using solve_ivp from scipy.integrate
    sol = integrate.solve_ivp(dydr, [R[-1], R[0]], y0, t_eval=R[::-1], method='RK45')
    
    # Extract the solution for u(r)
    u_r = sol.y[0][::-1]  # Reverse the solution to match the original order of R
    
    # Normalize the wave function using Simpson's rule
    norm_factor = integrate.simps(u_r**2, R)
    u_r_normalized = u_r / np.sqrt(norm_factor)
    
    return u_r_normalized

from scicode.parse.parse import process_hdf5_to_tuple
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
