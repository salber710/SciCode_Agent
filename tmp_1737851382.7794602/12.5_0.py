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
    if step <= 0:
        raise ValueError("Step size must be positive and non-zero.")

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


# Background: To solve the Schrödinger equation for a hydrogen-like atom, we can use the radial part of the equation
# and transform it into a form suitable for numerical methods. The Numerov method is particularly effective for solving
# second-order differential equations like the radial Schrödinger equation. The function f_Schrod computes the effective
# potential term f(r) for the radial equation. The Numerov method then uses this to compute the wavefunction u(r).
# After obtaining the wavefunction, it is important to normalize it to ensure that the total probability is 1.
# Simpson's rule is a numerical integration technique that can be used to compute the integral of the square of the
# wavefunction over the radial grid, which is necessary for normalization.

def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''


    # Define the effective potential function f_Schrod
    def f_Schrod(energy, l, r):
        # Corrected potential function for a hydrogen-like atom
        # V_eff = -1/r + l*(l+1)/(2*r^2) (in atomic units where Z=1 for hydrogen-like)
        # The potential energy part of the Schrödinger equation in atomic units
        return -1/r + l*(l+1)/(2*r**2) - energy

    # Define the Numerov method function
    def Numerov(f, u0, up0, h, r_grid):
        u = np.zeros_like(r_grid)
        u[0] = u0
        if len(r_grid) > 1:
            u[1] = u[0] + up0 * h
            for i in range(1, len(r_grid) - 1):
                h = r_grid[i+1] - r_grid[i]
                g0 = 1 + h**2 / 12 * f[i - 1]
                g1 = 1 + h**2 / 12 * f[i]
                g2 = 1 + h**2 / 12 * f[i + 1]
                u[i + 1] = (2 * u[i] * g1 - u[i - 1] * g0 + h**2 * f[i] * u[i]) / g2
        return u

    # Check for zero radius in r_grid to avoid division by zero in potential calculation
    if np.any(r_grid <= 0):
        raise ValueError("Radius grid contains zero or negative values, which may lead to division by zero in potential calculation.")

    # Calculate f(r) using the f_Schrod function
    f_r = f_Schrod(energy, l, r_grid)
    
    # Initial conditions for Numerov method
    u_at_0 = 0.0
    up_at_0 = -1e-7
    
    # Step size for Numerov method
    step = r_grid[1] - r_grid[0]  # Corrected to ensure positive step size
    
    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step, r_grid)
    
    # Normalize the wavefunction using Simpson's rule
    # We need to integrate |u(r)|^2 over the radial grid
    u_r_squared = u_r**2
    integral = integrate.simpson(u_r_squared, x=r_grid)  # Corrected to use keyword argument for x
    
    # Normalize u(r)
    ur_norm = u_r / np.sqrt(integral)
    
    return ur_norm


# Background: In quantum mechanics, the shooting method is a numerical technique used to solve boundary value problems,
# such as the Schrödinger equation. The method involves guessing an initial condition and integrating the differential
# equation to see if the boundary conditions are satisfied. If not, the initial condition is adjusted, and the process
# is repeated. In this context, we need to extrapolate the wavefunction at r=0 using the values at the first two points
# of the radial grid. This is done by dividing the wavefunction by r^l to account for the behavior of the wavefunction
# near the origin, where l is the angular momentum quantum number. The extrapolation is linear, using the first two
# points of the wavefunction divided by r^l.

def compute_Schrod(energy, r_grid, l):
    # Placeholder for the actual computation of the Schrödinger equation
    # This function should return a numpy array of the same length as r_grid
    # For simplicity, let's assume it returns the wavefunction values as r^2 * exp(-r) * energy

    return r_grid**2 * np.exp(-r_grid) * energy

def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''


    # Validate inputs
    if not isinstance(energy, (int, float)):
        raise TypeError("Energy must be a real number.")
    if not isinstance(l, int):
        raise TypeError("Angular momentum quantum number l must be an integer.")
    if np.any(np.isnan(r_grid)) or np.any(np.isinf(r_grid)):
        raise ValueError("r_grid must not contain NaN or infinite values.")
    if np.any(r_grid < 0):
        raise ValueError("r_grid must contain only non-negative values.")
    if len(r_grid) < 2:
        raise ValueError("r_grid must contain at least two elements.")

    # Compute the normalized wavefunction using the compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)
    
    # Divide the wavefunction by r^l, handling r=0 carefully
    with np.errstate(divide='ignore', invalid='ignore'):
        ur_divided = np.where(r_grid == 0, 0, ur_norm / (r_grid**l))
    
    # Linearly extrapolate the value at r=0 using the first two points
    # Linear extrapolation: f(0) = f(x1) - (f(x2) - f(x1)) * x1 / (x2 - x1)
    x1, x2 = r_grid[0], r_grid[1]
    f1, f2 = ur_divided[0], ur_divided[1]
    
    # Check for division by zero in case r_grid has identical values
    if x2 == x1:
        raise ValueError("r_grid must contain distinct values for extrapolation.")
    
    f_at_0 = f1 - (f2 - f1) * x1 / (x2 - x1)
    
    return f_at_0



# Background: In quantum mechanics, bound states refer to the discrete energy levels of a system where a particle is confined to a potential well. For a hydrogen-like atom, these are the energy levels where the electron is bound to the nucleus. The shooting method is a numerical technique used to find these bound states by solving the Schrödinger equation and adjusting the energy until the boundary conditions are satisfied. The `shoot` function is used to evaluate the wavefunction at the origin for a given energy, and a root-finding algorithm like Brent's method (`brentq` from `scipy.optimize`) is used to find the energies where the wavefunction satisfies the boundary conditions, indicating a bound state. The search is limited to a maximum of 10 bound states.



def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    
    def shoot_energy(energy):
        # Use the shoot function to find the wavefunction value at r=0 for a given energy
        return shoot(energy, r_grid, l)
    
    bound_states = []
    max_bound_states = 10
    
    # Iterate over the energy grid to find bound states
    for i in range(len(energy_grid) - 1):
        # Check if there is a sign change in the shoot function between two consecutive energies
        if shoot_energy(energy_grid[i]) * shoot_energy(energy_grid[i + 1]) < 0:
            # Use brentq to find the root (energy) where the wavefunction satisfies the boundary condition
            energy_bound = optimize.brentq(shoot_energy, energy_grid[i], energy_grid[i + 1])
            bound_states.append((l, energy_bound))
            
            # Stop if the maximum number of bound states is reached
            if len(bound_states) >= max_bound_states:
                break
    
    return bound_states

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.5', 3)
target = targets[0]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),2, -1.2/np.arange(1,20,0.2)**2), target)
target = targets[1]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),3,-1.2/np.arange(1,20,0.2)**2), target)
target = targets[2]

assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),0,-1.2/np.arange(1,20,0.2)**2), target)
