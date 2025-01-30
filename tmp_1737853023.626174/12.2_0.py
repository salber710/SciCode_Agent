from scipy import integrate
from scipy import optimize
import numpy as np

# Background: 
# The Schrödinger equation for a hydrogen-like atom can be expressed in spherical coordinates. 
# For a central potential, the radial part of the Schrödinger equation can be transformed into a form 
# that resembles a one-dimensional Schrödinger equation. This is done by substituting the wave function 
# with u(r) = r * R(r), where R(r) is the radial wave function. The resulting equation is:
# u''(r) = f(r) * u(r), where f(r) is a function of the potential energy, kinetic energy, and the 
# angular momentum quantum number l. 
# For the hydrogen atom, the potential energy is given by the Coulomb potential, and the effective 
# potential includes a centrifugal term due to the angular momentum. 
# The function f(r) can be derived from the radial part of the Schrödinger equation:
# f(r) = -2m/ħ^2 * (E + Z*e^2/(4πε₀r) - l(l+1)ħ^2/(2mr^2))
# In atomic units, where ħ = 1, m = 1, e = 1, and 4πε₀ = 1, this simplifies to:
# f(r) = -2 * (energy + 1/r - l*(l+1)/(2*r^2))
# Here, Z is set to 1 for the hydrogen atom.


def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    if not isinstance(energy, (int, float)):
        raise TypeError("Energy must be a numeric type.")
    if not isinstance(l, int):
        raise TypeError("Angular momentum quantum number l must be an integer.")
    if not isinstance(r_grid, np.ndarray):
        raise TypeError("r_grid must be a numpy array.")
    if np.any(r_grid <= 0):
        raise ValueError("Radius r must be positive to avoid division by zero or invalid values in the formula.")
    if l < 0:
        raise ValueError("Angular momentum quantum number l must be non-negative.")

    # Calculate f(r) for each r in the radial grid
    f_r = -2 * (energy + 1/r_grid - l*(l+1)/(2*r_grid**2))
    
    return f_r



# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form u''(r) = f(r)u(r). It is particularly useful for problems in quantum mechanics, such as solving the radial part of the Schrödinger equation. The method is based on a finite difference approach that provides a stable and accurate solution by considering terms up to the second derivative. The Numerov method uses a three-point recursion relation to compute the solution at each step, given initial conditions and a precomputed function f(r). The recursion relation is derived by expanding the function and its derivatives in a Taylor series and then eliminating higher-order terms to achieve higher accuracy.

def Numerov(f_in, u_at_0, up_at_0, step):
    '''Given precomputed function f(r), solve the differential equation u''(r) = f(r)*u(r)
    using the Numerov method.
    Inputs:
    - f_in: input function f(r); a 1D array of float representing the function values at discretized points.
    - u_at_0: the value of u at r = 0; a float.
    - up_at_0: the derivative of u at r = 0; a float.
    - step: step size; a float.
    Output:
    - u: the integration results at each point in the radial grid; a 1D array of float.
    '''
    # Number of points in the grid
    n_points = len(f_in)
    
    # Initialize the solution array
    u = np.zeros(n_points)
    
    # Set initial conditions
    u[0] = u_at_0
    u[1] = u_at_0 + step * up_at_0 + (step**2 / 2) * f_in[0] * u_at_0
    
    # Numerov's method iteration
    for i in range(1, n_points - 1):
        u[i + 1] = (2 * (1 - (5/12) * step**2 * f_in[i]) * u[i] - 
                    (1 + (1/12) * step**2 * f_in[i - 1]) * u[i - 1]) / (1 + (1/12) * step**2 * f_in[i + 1])
    
    return u

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.2', 3)
target = targets[0]

assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[1]

assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
target = targets[2]

assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)
