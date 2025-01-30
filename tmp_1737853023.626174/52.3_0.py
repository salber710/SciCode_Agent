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



def Schroed_deriv(y, r, l, En):
    """ Placeholder for the actual Schroedinger derivative function. """
    # This is a mock-up function. In practice, this would compute the derivatives of the wave function.
    u, uprime = y
    V = l * (l + 1) / r**2  # Simplified potential term for a radial quantum problem
    dudr = uprime
    duprimedr = (2.0 / r) * uprime - (2 * (En - V)) * u
    return [dudr, duprimedr]

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
    # Validate input R for negative or zero values which are not physical in this context
    if np.any(R <= 0):
        raise ValueError("Radius R must contain only positive values.")

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



# Background: 
# In quantum mechanics, the wave function must be well-behaved at the origin (r=0). 
# For the radial wave function of a hydrogen-like atom, the behavior near the origin is influenced by the angular momentum quantum number `l`.
# Specifically, the wave function `u(r)` behaves as `r^l` near the origin. 
# To estimate the value of the wave function at r=0, we can use linear extrapolation based on the values at the first two grid points.
# Before performing the extrapolation, we divide the wave function values by `r^l` to account for the expected behavior near the origin.
# This approach is part of the shooting method, which is used to find the energy levels by matching boundary conditions.

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
    # Solve the Schrödinger equation to get the wave function values
    u_r_normalized = SolveSchroedinger(y0, En, l, R)
    
    # Get the first two points in the radial grid
    r1, r2 = R[0], R[1]
    u1, u2 = u_r_normalized[0], u_r_normalized[1]
    
    # Divide the wave function values by r^l to account for the behavior near the origin
    f1 = u1 / (r1**l)
    f2 = u2 / (r2**l)
    
    # Perform linear extrapolation to estimate the value at r=0
    # Using the formula for linear extrapolation: f(0) = f1 + (f2 - f1) * (0 - r1) / (r2 - r1)
    f_at_0 = f1 + (f2 - f1) * (0 - r1) / (r2 - r1)
    
    return f_at_0

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('52.3', 3)
target = targets[0]

assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)
target = targets[1]

assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)
target = targets[2]

assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)
