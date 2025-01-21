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


# Background: The Schrödinger equation describes quantum mechanical systems. For hydrogen-like atoms, the radial part of 
# the wave function can be computed by solving a second-order differential equation using f_Schrod to calculate the function 
# f(r) and the Numerov method to compute the radial wavefunction u(r). After computing u(r), it is essential to normalize the 
# wave function to ensure that its probability interpretation holds. Normalization ensures the total probability of finding 
# the electron in space sums to one. Simpson's rule is a numerical integration technique that provides an efficient way to 
# compute definite integrals, which can be used here to normalize the wave function over the given radial grid.

def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''


    # Calculate the function f(r) using the f_Schrod function
    f_r = f_Schrod(energy, l, r_grid)

    # Define initial conditions for Numerov method
    u_at_0 = 0.0
    up_at_0 = -1e-7

    # Calculate the step size for the radial grid
    step = r_grid[0] - r_grid[1]

    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step)

    # Normalize u(r) using Simpson's rule over the radial grid
    # Simpson's rule for integration requires the wavefunction squared
    probability_density = u_r**2

    # Compute the integral using Simpson's rule over the radial grid
    integral = integrate.simpson(probability_density, r_grid)

    # Normalize the wave function
    ur_norm = u_r / np.sqrt(integral)

    return ur_norm



# Background: In quantum mechanics, especially when dealing with problems like the hydrogen atom, the radial wave function 
# is often expressed in terms of u(r) and involves solving the Schrödinger equation. In computational approaches like 
# the shooting method, we often need to evaluate the behavior of the wavefunction at the origin (r=0) to match boundary 
# conditions. Since the wavefunction can be singular or undefined at r=0 due to division by zero issues, we use 
# extrapolation techniques. Specifically, we employ linear extrapolation using the values of the wavefunction at nearby 
# points. Before extrapolating, we divide the wavefunction by r^l to account for the angular momentum barrier in the 
# Schrödinger equation, which modifies the behavior of the wavefunction near the origin.

def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''
    # Calculate the wavefunction using the compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)

    # Divide the wavefunction by r**l to handle the angular momentum term
    ur_norm_divided = ur_norm / r_grid**l

    # Perform linear extrapolation to estimate the wavefunction at r=0
    # Using the first two points r_grid[0] and r_grid[1]
    r1 = r_grid[0]
    r2 = r_grid[1]
    u1 = ur_norm_divided[0]
    u2 = ur_norm_divided[1]

    # Linear extrapolation formula: u_at_0 = u1 + (0 - r1) * (u2 - u1) / (r2 - r1)
    f_at_0 = u1 + (-r1) * (u2 - u1) / (r2 - r1)

    return f_at_0

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.4', 3)
target = targets[0]

assert np.allclose(shoot(1.1, np.linspace(1e-7,20,10), 0), target)
target = targets[1]

assert np.allclose(shoot(1.1, np.linspace(1e-7,20,10), 1), target)
target = targets[2]

assert np.allclose(shoot(1.1, np.linspace(1e-7,50,10), 2), target)
