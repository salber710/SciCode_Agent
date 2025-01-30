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
    # Calculate the second derivative of u using the radial Schrödinger equation
    u_double_prime = (l * (l + 1) / r**2 - 2 / r + 2 * En) * u
    # Return the derivatives as a numpy array
    Schroed = np.array([u_prime, u_double_prime])
    return Schroed

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.1', 3)
target = targets[0]

y = [0.0,-1e-5]
r = 100
l = 1
En = 1
assert np.allclose(Schroed_deriv(y,r,l,En), target)
target = targets[1]

y = [0.0,-2e-5]
r = 1.1
l = 2
En = 1.5
assert np.allclose(Schroed_deriv(y,r,l,En), target)
target = targets[2]

y = [0.0,-2e-5]
r = 3
l = 1
En = 5
assert np.allclose(Schroed_deriv(y,r,l,En), target)
