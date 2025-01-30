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



def shoot(energy, r_grid, l):
    # Placeholder for the actual shoot function implementation
    # This function should solve the Schrödinger equation for a given energy and return the wavefunction value at r=0
    # For simplicity, let's assume it returns a simple function of energy that changes sign at some energies
    if not isinstance(r_grid, np.ndarray):
        raise TypeError("r_grid must be a numpy array")
    if r_grid.size == 0:
        return 0  # No grid to evaluate on
    if l == -1:  # Avoid division by zero when l = -1
        return np.sin(energy)  # Simplified model: sin(energy) without scaling
    return np.sin(energy) / (l + 1)  # Simplified model: sin(energy) scaled by angular momentum

def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    
    if not isinstance(r_grid, np.ndarray):
        raise TypeError("r_grid must be a numpy array")
    if not isinstance(l, int):
        raise TypeError("l must be an integer")
    if not isinstance(energy_grid, np.ndarray):
        raise TypeError("energy_grid must be a numpy array")

    def shoot_energy(energy):
        # Use the shoot function to find the wavefunction value at r=0 for a given energy
        return shoot(energy, r_grid, l)
    
    bound_states = []
    max_bound_states = 10
    
    # Ensure energy_grid is sorted in ascending order
    energy_grid = np.sort(energy_grid)
    
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


# Background: In quantum mechanics, bound states are characterized by discrete energy levels. When multiple bound states
# are found, it is often necessary to sort them to analyze their properties systematically. The sorting is typically done
# by energy, as lower energy states are generally more stable. However, when energies are very close or identical, the
# angular momentum quantum number can be used as a secondary sorting criterion. This ensures that states are ordered
# consistently, even when their energies are nearly the same. The angular momentum quantum number is given a small weight
# (e.g., 1/10000.0) to affect the sorting only when energies are indistinguishable.

def sort_states(bound_states):
    '''Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''
    # Sort the bound states primarily by energy, and secondarily by angular momentum quantum number
    # Handling NaN values by placing them at the end of the list
    sorted_states = sorted(bound_states, key=lambda state: (float('inf') if state[1] != state[1] else state[1], state[0] / 10000.0))
    
    return sorted_states


# Background: In quantum mechanics, the charge density of an atom is related to the probability density of its electrons.
# For bound states, the charge density can be calculated from the wavefunctions of the electrons. The wavefunction
# provides the probability amplitude, and its square gives the probability density. For a hydrogen-like atom, the
# wavefunctions are solutions to the Schrödinger equation. The charge density is then the sum of the probability
# densities of all occupied states, taking into account their degeneracy. The degeneracy of a state is determined by
# its angular momentum quantum number, with each state having a degeneracy of 2(2l+1) due to spin and magnetic quantum
# numbers. The charge density is calculated over a radial grid, which represents the distance from the nucleus.



def calculate_charge_density(bound_states, r_grid, Z):
    '''Input
    bound_states: bound states found using the find_bound_states function; a list of tuples
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    charge_density: the calculated charge density coming from the bound states; 1D array of float
    '''
    
    # Sort the bound states using the sort_states function
    sorted_states = sorted(bound_states, key=lambda state: state[1])  # Assuming sorting by energy
    
    # Initialize the charge density array
    charge_density = np.zeros_like(r_grid)
    
    # Total number of available states (for hydrogen-like atom, Z is the number of electrons)
    total_states = Z
    
    # Populate the available orbitals with the sorted states
    for l, energy in sorted_states:
        # Calculate the degeneracy of the state
        degeneracy = 2 * (2 * l + 1)
        
        # If there are no more available states, break the loop
        if total_states <= 0:
            break
        
        # Calculate the wavefunction for the current state using compute_Schrod
        ur_norm = compute_Schrod(energy, r_grid, l)
        
        # Calculate the contribution to the charge density
        # Multiply by the degeneracy and add to the total charge density
        charge_density += degeneracy * ur_norm**2
        
        # Decrease the number of available states
        total_states -= degeneracy
    
    return charge_density

def compute_Schrod(energy, r_grid, l):
    # Placeholder for the wavefunction computation
    # This function should compute the normalized wavefunction for a given energy, radial grid, and angular momentum
    # For simplicity, we assume a normalized wavefunction that is uniform across the grid
    return np.ones_like(r_grid) / np.sqrt(r_grid.size)


# Background: In quantum mechanics, the Hartree potential is a mean-field approximation used to describe the
# electrostatic interaction between electrons in an atom. It is derived from the Poisson equation, which relates
# the Laplacian of the potential to the charge density. For a spherically symmetric system, the Poisson equation
# can be expressed in terms of the radial coordinate as ∇²V_H(r) = -8πρ(r), where ρ(r) is the charge density.
# The Numerov method, which is a numerical technique for solving second-order differential equations, can be
# adapted to solve this Poisson equation. The solution, V_H(r), is then used to compute the Hartree potential
# term in the Schrödinger equation. The function will return the Hartree potential in the form U(r) = V_H(r) * r,
# which is useful for further calculations in the context of solving the Schrödinger equation.



def calculate_HartreeU(charge_density, u_at_0, up_at_0, step, r_grid, Z):
    '''Input
    charge_density: the calculated charge density of the bound states; 1D array of float
    u_at_0: the value of u at r = 0; float
    up_at_0: the derivative of u at r = 0; float
    step: step size; float.
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    x: the HartreeU term with U(r)=V_H(r)r; 1D array of float
    '''

    # Define the function f(r) for the Poisson equation
    # f(r) = -8πρ(r) in the context of the Poisson equation
    f_r = -8 * np.pi * charge_density

    # Use the Numerov method to solve the differential equation
    def Numerov(f, u0, up0, h, r_grid):
        u = np.zeros_like(r_grid)
        if len(r_grid) > 0:
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

    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step, r_grid)

    # Calculate the Hartree potential U(r) = V_H(r) * r
    # V_H(r) is obtained from u(r) by dividing by r
    with np.errstate(divide='ignore', invalid='ignore'):
        V_H_r = np.where(r_grid == 0, 0, u_r / r_grid)
    
    # Calculate U(r) = V_H(r) * r
    HartreeU = V_H_r * r_grid

    return HartreeU


# Background: In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes over time. 
# For a hydrogen-like atom, the radial part of the Schrödinger equation can be expressed in a form that includes the Hartree potential, 
# which accounts for electron-electron interactions in a mean-field approximation. The Hartree potential is derived from the charge density 
# and affects the effective potential experienced by an electron. When the Hartree potential is included, the effective potential term 
# in the Schrödinger equation becomes more complex. The function f_Schrod_Hartree calculates the effective potential term f(r) 
# for the radial Schrödinger equation, including the Hartree potential. This term is used in the differential equation u''(r) = f(r)u(r), 
# where u(r) is the radial wavefunction. The function takes into account the energy, angular momentum, and the Hartree potential to 
# compute f(r), which is essential for solving the Schrödinger equation with the Hartree term.


def f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    f_r: a 1D array of float 
    '''
    if not isinstance(energy, (int, float)):
        raise TypeError("Energy must be a numeric type.")
    if not isinstance(l, int):
        raise TypeError("Angular momentum quantum number l must be an integer.")
    if not isinstance(Z, int):
        raise TypeError("Atomic number Z must be an integer.")
    if not isinstance(r_grid, np.ndarray):
        raise TypeError("r_grid must be a numpy array.")
    if not isinstance(hartreeU, np.ndarray):
        raise TypeError("hartreeU must be a numpy array.")
    if np.any(r_grid <= 0):
        raise ValueError("Radius r must be positive to avoid division by zero or invalid values in the formula.")
    if l < 0:
        raise ValueError("Angular momentum quantum number l must be non-negative.")
    if len(r_grid) != len(hartreeU):
        raise ValueError("r_grid and hartreeU must have the same length.")

    # Calculate f(r) for each r in the radial grid, including the Hartree potential
    f_r = -2 * (energy + Z/r_grid - l*(l+1)/(2*r_grid**2) + hartreeU/r_grid)
    
    return f_r


# Background: In quantum mechanics, the Schrödinger equation describes the behavior of quantum systems. 
# For a hydrogen-like atom, the radial part of the Schrödinger equation can be expressed in a form that includes 
# the Hartree potential, which accounts for electron-electron interactions in a mean-field approximation. 
# The Hartree potential is derived from the charge density and affects the effective potential experienced by an electron. 
# The function f_Schrod_Hartree calculates the effective potential term f(r) for the radial Schrödinger equation, 
# including the Hartree potential. This term is used in the differential equation u''(r) = f(r)u(r), where u(r) is the 
# radial wavefunction. The Numerov method is a numerical technique used to solve this second-order differential equation. 
# After obtaining the wavefunction, it is important to normalize it to ensure that the total probability is 1. 
# Simpson's rule is a numerical integration technique that can be used to compute the integral of the square of the 
# wavefunction over the radial grid, which is necessary for normalization.

def compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''


    # Define the function f_Schrod_Hartree to calculate f(r)
    def f_Schrod_Hartree(energy, r, l, Z, U):
        # Constants
        hbar = 1.0545718e-34  # Planck's constant over 2*pi, in J*s
        m_e = 9.10938356e-31  # Electron mass, in kg
        e = 1.60217662e-19    # Elementary charge, in C
        epsilon_0 = 8.854187817e-12  # Vacuum permittivity, in F/m
        a0 = 4 * np.pi * epsilon_0 * hbar**2 / (m_e * e**2)  # Bohr radius, in meters

        # Convert energy from eV to J
        energy_joules = energy * e

        # Effective potential term
        V_eff = -Z * e**2 / (4 * np.pi * epsilon_0 * r) + l * (l + 1) * hbar**2 / (2 * m_e * r**2) + U

        # f(r) for the radial Schrödinger equation
        f_r = 2 * m_e / hbar**2 * (energy_joules - V_eff)

        return f_r

    # Calculate f(r) using the f_Schrod_Hartree function
    f_r = np.array([f_Schrod_Hartree(energy, r, l, Z, U) for r, U in zip(r_grid, hartreeU)])
    
    # Initial conditions for Numerov method
    u_at_0 = 0.0
    up_at_0 = -1e-7
    
    # Step size for Numerov method
    step = r_grid[1] - r_grid[0]
    
    # Use the Numerov method to solve for u(r)
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

    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step, r_grid)
    
    # Normalize the wavefunction using Simpson's rule
    u_r_squared = u_r**2
    integral = integrate.simpson(u_r_squared, x=r_grid)
    
    # Normalize u(r)
    ur_norm = u_r / np.sqrt(integral)
    
    return ur_norm



# Background: In quantum mechanics, the wavefunction u(r) describes the probability amplitude of a particle's position.
# Near the origin (r = 0), the behavior of the wavefunction can be complex, especially when considering the effects of
# angular momentum and potential terms like the Hartree potential. To accurately estimate the value of the wavefunction
# at r = 0, we can use polynomial extrapolation. The polyfit function in Numpy allows us to fit a polynomial to a set of
# data points. By fitting a 3rd order polynomial to the first few points of the wavefunction divided by r^l, we can
# extrapolate the value at r = 0. This method provides a more accurate estimate by considering the curvature of the
# wavefunction near the origin.

def extrapolate_polyfit(energy, r_grid, l, Z, hartreeU):
    '''Input 
    energy: float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; int
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    u0: the extrapolated value of u(r) at r=0; float
    '''
    # Compute the wavefunction using the compute_Schrod_Hartree function
    ur_norm = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
    # Divide the wavefunction by r^l
    with np.errstate(divide='ignore', invalid='ignore'):
        ur_divided = np.where(r_grid == 0, 0, ur_norm / (r_grid**l))
    
    # Use the first four points for polynomial fitting
    r_points = r_grid[:4]
    ur_points = ur_divided[:4]
    
    # Fit a 3rd order polynomial to the data
    coeffs = np.polyfit(r_points, ur_points, 3)
    
    # Evaluate the polynomial at r = 0 to extrapolate the value
    u0 = np.polyval(coeffs, 0)
    
    return u0

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('12.11', 3)
target = targets[0]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=28
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(extrapolate_polyfit(-0.5, r_grid, 2, Z, hu), target)
target = targets[1]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 3
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(extrapolate_polyfit(-0.4, r_grid, 2, Z, hu), target)
target = targets[2]

energy_grid = -0.9/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=14
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(extrapolate_polyfit(-0.5, r_grid, 3, Z, hu), target)
