import numpy as np
from scipy import integrate, optimize

# Background: 
# The radial part of the Schrödinger equation for a hydrogen-like atom can be expressed in terms of the radial wave function u(r), 
# which is related to the actual wave function R(r) by u(r) = r * R(r). The equation is given by:
# 
# d^2u/dr^2 = [l(l+1)/r^2 - 2Z/r + 2En] * u
# 
# where l is the angular momentum quantum number, Z is the atomic number (Z=1 for hydrogen), and En is the energy of the state.
# This second-order differential equation can be rewritten as a system of first-order differential equations:
# 
# Let y = [u, u'], where u' = du/dr. Then:
# dy/dr = [u', u'']
# 
# where u'' = [l(l+1)/r^2 - 2Z/r + 2En] * u.
# 
# The function Schroed_deriv calculates the derivative of y with respect to r, given the current values of y, r, l, and En.

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
    Z = 1  # Atomic number for hydrogen
    u, u_prime = y  # Unpack the current values of u and u'
    
    # Calculate the second derivative of u using the radial Schrödinger equation
    u_double_prime = (l * (l + 1) / r**2 - 2 * Z / r + 2 * En) * u
    
    # Return the derivatives as a numpy array
    Schroed = np.array([u_prime, u_double_prime])
    
    return Schroed



# Background: 
# To solve the radial part of the Schrödinger equation for a hydrogen-like atom, we need to integrate the system of first-order differential equations
# derived from the second-order differential equation. The function Schroed_deriv provides the derivatives needed for this integration.
# We will use numerical integration to solve this system over a range of radii, starting from a large radius where the wave function is expected to be small.
# After obtaining the solution, we need to normalize the wave function. Simpson's rule is a numerical method that can be used to perform this normalization
# by integrating the square of the wave function over the range of interest.



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
    # Integrate the system of differential equations using solve_ivp
    sol = integrate.solve_ivp(
        fun=lambda r, y: Schroed_deriv(y, r, l, En),
        t_span=(R[0], R[-1]),
        y0=y0,
        t_eval=R,
        method='RK45'
    )
    
    # Extract the solution for u(r)
    u_r = sol.y[0]
    
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
