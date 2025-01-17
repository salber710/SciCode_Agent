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



# Background: 
# In quantum mechanics, the wave function of a particle provides information about the probability amplitude of its position and momentum.
# For a hydrogen-like atom, the radial wave function u(r) is often solved numerically over a range of radii. 
# However, the value of the wave function at r=0 is not directly computed in numerical solutions because the radial grid typically starts at a small positive value.
# To estimate the wave function at r=0, we can use linear extrapolation based on the values at the first two grid points.
# Before performing the extrapolation, the wave function values are divided by r^l to account for the behavior of the wave function near the origin,
# where l is the angular momentum quantum number. This division helps in obtaining a more accurate extrapolation by considering the expected behavior of the wave function.

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
    
    # Divide the wave function values by r^l
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
