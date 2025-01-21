from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np

# Background: 
# The Schrödinger equation in quantum mechanics describes how the quantum state of a physical system changes over time. 
# For a central potential problem, such as the hydrogen atom, the radial part of the wave function satisfies a differential equation.
# When rewritten, the radial Schrödinger equation can be transformed into a form involving a radial function u(r), where
# u''(r) = f(r) * u(r). The potential involves a Coulomb potential term along with a kinetic term. In atomic units, these terms 
# simplify considerably. For hydrogen-like atoms with Z = 1, the potential term is -1/r. The kinetic term involves the second 
# derivative which is related to the angular momentum quantum number l. The function f(r) encapsulates these terms as:
# f(r) = 2 * (energy + 1/r) - l*(l+1)/r^2. This function f(r) is crucial for solving the radial Schrödinger equation using
# numerical methods.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Calculate f(r) using the relation derived from the radial Schrödinger equation
    # f(r) = 2 * (energy + 1/r) - l*(l+1)/r^2
    # We assume Z = 1 for hydrogen-like atoms

    # Pre-allocate the output array
    f_r = np.zeros_like(r_grid)

    # Calculate f(r) for each value of r in r_grid
    for i, r in enumerate(r_grid):
        if r != 0:
            f_r[i] = 2 * (energy + 1/r) - l * (l + 1) / r**2
        else:
            f_r[i] = float('inf')  # Handle the singularity at r = 0

    return f_r


# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form
# u''(r) = f(r)u(r). It is particularly useful for equations where the second derivative of u is expressed as a product of u and 
# a function f(r), as is the case with the radial part of the Schrödinger equation for hydrogen-like atoms. The method is 
# efficient and accurate for such problems because it incorporates information from multiple points in the discretized domain, 
# leading to higher-order accuracy. The method requires initial values of the function and its first derivative and uses a 
# recurrence relation to compute subsequent values. The recurrence relation for the Numerov method is given by:
# u_{n+1} = (2u_n(1 - 5h^2f_n/12) - u_{n-1}(1 + h^2f_{n-1}/12) + h^2f_{n+1}u_{n+1}/12) / (1 + h^2f_{n+1}/12),
# where h is the step size and f_n is the value of the function f(r) at the nth point.

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
    num_points = len(f_in)

    # Pre-allocate the output array for u
    u = np.zeros(num_points)

    # Initial conditions
    u[0] = u_at_0
    # Calculate u[1] using the Taylor expansion approximation
    u[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * f_in[0] * u_at_0

    # Apply the Numerov method
    for n in range(1, num_points - 1):
        # Calculate the next value of u using the Numerov recurrence relation
        u[n + 1] = ((2 * u[n] * (1 - (5/12) * step**2 * f_in[n]) -
                     u[n - 1] * (1 + (1/12) * step**2 * f_in[n - 1])) /
                    (1 + (1/12) * step**2 * f_in[n + 1]))

    return u



# Background: To solve the Schrödinger equation for the radial part of a hydrogen-like atom, we use the Numerov method 
# to integrate the differential equation u''(r) = f(r)u(r), where f(r) is determined by the energy and angular momentum 
# quantum number. After obtaining the numerical solution for the radial wave function u(r) on a grid, it is necessary 
# to normalize the wave function. Normalization ensures that the probability of finding the electron in the entire space 
# is equal to one. Simpson's rule, a numerical integration method, can be used for this purpose. The step size for the 
# Numerov method is calculated as the difference between the two largest radii in the grid, and initial conditions are 
# set such that u_at_0 is 0 and up_at_0 is a small negative number to avoid singular behavior at the origin.

def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''

    # Calculate f(r) using the provided f_Schrod function
    f_r = f_Schrod(energy, l, r_grid)

    # Initial conditions for the Numerov method
    u_at_0 = 0.0
    up_at_0 = -1e-7
    step = r_grid[0] - r_grid[1]

    # Solve for u(r) using the Numerov method
    u = Numerov(f_r, u_at_0, up_at_0, step)

    # Normalize u(r) using Simpson's rule
    # We need to integrate |u|^2 over the radial grid
    u_squared = u**2

    # Compute the normalization factor using Simpson's rule
    normalization_factor = integrate.simps(u_squared, r_grid)

    # Normalize u(r)
    ur_norm = u / np.sqrt(normalization_factor)

    return ur_norm


try:
    targets = process_hdf5_to_tuple('12.3', 3)
    target = targets[0]
    assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)

    target = targets[1]
    assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)

    target = targets[2]
    assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e