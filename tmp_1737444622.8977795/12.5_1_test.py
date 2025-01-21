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


# Background: In quantum mechanics, the wavefunction describes the quantum state of a particle. For the radial part of
# the wavefunction of hydrogen-like atoms, it is necessary to perform certain transformations for numerical stability
# and accuracy. When dealing with angular momentum (characterized by the quantum number l), the radial wavefunction u(r)
# is often divided by r^l to simplify the boundary conditions at r = 0. In numerical simulations, especially when using
# the shooting method to solve boundary value problems, it is common to use linear extrapolation to estimate values at
# boundaries. Here, the task is to extrapolate the wavefunction at r = 0 using the first two points of the numerically 
# computed wavefunction u(r) transformed by dividing by r^l. This is part of preparing the wavefunction for further 
# computational steps, such as matching boundary conditions in the shooting method.

def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''

    # Compute the normalized wavefunction using the compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)

    # Divide the wavefunction by r^l for each point in the radial grid
    ur_transformed = ur_norm / (r_grid**l)

    # Perform linear extrapolation using the first two values of ur_transformed
    # ur_transformed[0] corresponds to u(r) / r^l at the first grid point
    # ur_transformed[1] corresponds to u(r) / r^l at the second grid point
    # Linear extrapolation formula: f_at_0 = ur_transformed[0] + (ur_transformed[1] - ur_transformed[0]) * (0 - r_grid[0]) / (r_grid[1] - r_grid[0])

    # Calculate the linear extrapolation at r=0
    f_at_0 = ur_transformed[0] + (ur_transformed[1] - ur_transformed[0]) * (0 - r_grid[0]) / (r_grid[1] - r_grid[0])

    return f_at_0



# Background: In quantum mechanics, bound states are states where a particle is confined to a finite region of space 
# and has an energy less than the potential at infinity. For hydrogen-like atoms, bound states correspond to discrete 
# energy levels where the electron is bound to the nucleus. To find these energy levels numerically, we can use 
# the shooting method, which involves guessing an energy, solving the Schrödinger equation, and checking the boundary 
# conditions. The shoot function helps extrapolate the wavefunction at the origin using the computed wavefunction at 
# two nearby points. The root-finding routine like brentq in scipy can be used to find energies where the extrapolated 
# wavefunction satisfies the boundary condition (typically zero at the origin). By scanning over a range of energies 
# and using brentq, which requires a sign change over an interval to detect a root, we can identify the energies 
# corresponding to bound states. The search is limited to a maximum of 10 bound states to ensure computational efficiency.

def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''

    # List to store found bound states
    bound_states = []

    # Iterate over the input energy grid to find bound states
    for i in range(len(energy_grid) - 1):
        e1 = energy_grid[i]
        e2 = energy_grid[i + 1]

        # Check if there is a sign change in the shoot function's output
        try:
            # Use the brentq root-finding method to find the energy where the wavefunction is zero at r=0
            bound_energy = optimize.brentq(lambda energy: shoot(energy, r_grid, l), e1, e2)
            bound_states.append((l, bound_energy))

            # Limit the number of bound states to 10
            if len(bound_states) >= 10:
                break
        except ValueError:
            # If brentq fails, it means no root is found in this interval
            continue

    return bound_states

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.5', 3)
target = targets[0]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),2, -1.2/np.arange(1,20,0.2)**2), target)
target = targets[1]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),3,-1.2/np.arange(1,20,0.2)**2), target)
target = targets[2]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),0,-1.2/np.arange(1,20,0.2)**2), target)
