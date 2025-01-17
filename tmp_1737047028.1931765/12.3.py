from scipy import integrate
from scipy import optimize
import numpy as np

# Background: 
# The Schrödinger equation for a hydrogen-like atom can be expressed in spherical coordinates. 
# For the radial part, the equation can be transformed into a form involving the radial wave function u(r), 
# where u(r) = r * R(r) and R(r) is the radial part of the wave function. 
# The transformed equation is u''(r) = f(r)u(r), where f(r) is a function that depends on the potential energy, 
# the angular momentum quantum number l, and the energy of the system.
# The potential energy term for a hydrogen-like atom is given by the Coulomb potential, 
# V(r) = -Ze^2 / (4πε₀r), where Z is the atomic number (Z=1 for hydrogen), e is the elementary charge, 
# and ε₀ is the permittivity of free space. 
# The effective potential also includes a centrifugal term due to the angular momentum, 
# which is l(l+1)ħ²/(2mr²), where ħ is the reduced Planck's constant and m is the electron mass.
# The function f(r) is derived from these terms and the given energy E.

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J·s
    m_e = 9.10938356e-31  # Electron mass in kg
    e = 1.60217662e-19    # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    Z = 1  # Atomic number for hydrogen

    # Precompute constants
    hbar2_over_2m = hbar**2 / (2 * m_e)
    Ze2_over_4pi_epsilon0 = Z * e**2 / (4 * np.pi * epsilon_0)

    # Calculate f(r) for each r in r_grid
    f_r = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r == 0:
            # Avoid division by zero; handle r = 0 case separately if needed
            f_r[i] = 0
        else:
            # Effective potential term
            V_eff = -Ze2_over_4pi_epsilon0 / r + l * (l + 1) * hbar2_over_2m / r**2
            # f(r) = (2m/ħ²) * (V_eff - E)
            f_r[i] = (2 * m_e / hbar**2) * (V_eff - energy)

    return f_r


# Background: The Numerov method is a numerical technique used to solve second-order linear differential equations of the form u''(r) = f(r)u(r). 
# It is particularly useful for problems in quantum mechanics, such as solving the radial part of the Schrödinger equation. 
# The method is based on a finite difference approach that provides a stable and accurate solution by considering terms up to the second derivative. 
# The Numerov method uses a three-point recursion relation to compute the solution at each step, given the initial conditions u(0) and u'(0), 
# and the precomputed function f(r). The recursion relation is derived by expanding the function in a Taylor series and matching terms.

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
    if n_points > 1:
        # Use the Taylor expansion to estimate the second point
        u[1] = u_at_0 + step * up_at_0 + (step**2 / 2) * f_in[0] * u_at_0
    
    # Numerov's method coefficients
    k = (step**2) / 12.0
    
    # Apply the Numerov method
    for i in range(1, n_points - 1):
        u[i + 1] = (2 * (1 - 5 * k * f_in[i]) * u[i] - (1 + k * f_in[i - 1]) * u[i - 1]) / (1 + k * f_in[i + 1])
    
    return u



# Background: To solve the Schrödinger equation for a hydrogen-like atom, we use the radial part of the equation, which can be transformed into a second-order differential equation of the form u''(r) = f(r)u(r). The function f(r) is derived from the potential energy and the angular momentum quantum number. The Numerov method is a numerical technique used to solve such differential equations. It provides a stable and accurate solution by using a three-point recursion relation. After solving for the radial wave function u(r), it is important to normalize the wave function. Normalization ensures that the total probability of finding the electron in space is 1. Simpson's rule is a numerical integration method that can be used to perform this normalization.

def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''


    # Calculate f(r) using the f_Schrod function
    f_r = f_Schrod(energy, l, r_grid)

    # Initial conditions
    u_at_0 = 0
    up_at_0 = -1e-7

    # Numerov step size
    step = r_grid[0] - r_grid[1]

    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step)

    # Normalize u(r) using Simpson's rule
    # We integrate from the largest radius, so reverse the grid and u_r
    r_grid_reversed = r_grid[::-1]
    u_r_reversed = u_r[::-1]

    # Calculate the integral of |u(r)|^2 over r using Simpson's rule
    integral = integrate.simps(u_r_reversed**2, r_grid_reversed)

    # Normalize u(r)
    ur_norm = u_r / np.sqrt(integral)

    return ur_norm


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('12.3', 3)
target = targets[0]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)
target = targets[1]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)
target = targets[2]

assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)
