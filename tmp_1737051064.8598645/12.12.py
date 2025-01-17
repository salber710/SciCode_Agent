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


# Background: In quantum mechanics, the shooting method is a numerical technique used to solve boundary value problems, such as the Schrödinger equation. 
# The method involves guessing an initial condition, solving the differential equation, and then adjusting the guess based on the boundary conditions. 
# In this context, we need to extrapolate the value of the wavefunction at r=0 using the values at the first two grid points. 
# This is done by dividing the wavefunction by r^l, where r is the radius and l is the angular momentum quantum number, to account for the behavior of the wavefunction near the origin. 
# Linear extrapolation is then used to estimate the wavefunction value at r=0.

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
    
    # Divide the wavefunction by r^l
    ur_divided = ur_norm / (r_grid**l)
    
    # Perform linear extrapolation to estimate the value at r=0
    # Using the first two points in the radial grid
    r1, r2 = r_grid[0], r_grid[1]
    u1, u2 = ur_divided[0], ur_divided[1]
    
    # Linear extrapolation formula: f(0) = f(x1) + (f(x2) - f(x1)) * (0 - x1) / (x2 - x1)
    f_at_0 = u1 + (u2 - u1) * (0 - r1) / (r2 - r1)
    
    return f_at_0


# Background: In quantum mechanics, bound states refer to the discrete energy levels of a system where a particle is confined to a particular region of space, such as an electron in an atom. 
# For a hydrogen-like atom, these bound states correspond to negative energy solutions of the Schrödinger equation. 
# The shooting method, combined with a root-finding algorithm, can be used to find these bound states by searching for energies where the wavefunction satisfies the boundary conditions. 
# The brentq method from scipy is a robust root-finding algorithm that can be used to find the zeros of a function within a specified interval. 
# By applying the shoot function to different energies and using brentq to find where the wavefunction at r=0 is zero, we can identify the bound states.



def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    bound_states = []
    max_bound_states = 10
    num_found = 0

    # Iterate over the energy grid to find bound states
    for i in range(len(energy_grid) - 1):
        if num_found >= max_bound_states:
            break

        # Define the function to find the root of
        def shoot_energy(energy):
            return shoot(energy, r_grid, l)

        # Check if there is a sign change between energy_grid[i] and energy_grid[i+1]
        if shoot_energy(energy_grid[i]) * shoot_energy(energy_grid[i + 1]) < 0:
            # Use brentq to find the root (bound state energy)
            try:
                bound_energy = optimize.brentq(shoot_energy, energy_grid[i], energy_grid[i + 1])
                bound_states.append((l, bound_energy))
                num_found += 1
            except ValueError:
                # If brentq fails, continue to the next interval
                continue

    return bound_states


# Background: In quantum mechanics, bound states are characterized by discrete energy levels. 
# When multiple bound states are found, it is often necessary to sort them for analysis or further computation. 
# The sorting is typically done based on energy, as lower energy states are generally more stable and of greater interest. 
# However, when energies are very close or identical, additional criteria such as the angular momentum quantum number can be used to break ties. 
# In this context, we sort bound states primarily by energy, and secondarily by angular momentum quantum number, 
# with a small weight to ensure that angular momentum only affects the order when energies are nearly identical.

def sort_states(bound_states):
    '''Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''
    # Sort the bound states by energy first, and then by angular momentum quantum number
    # Use a small factor to ensure angular momentum affects the order only when energies are very close
    sorted_states = sorted(bound_states, key=lambda state: (state[1], state[0] / 10000.0))
    
    return sorted_states


# Background: In quantum mechanics, the charge density of an atom is related to the probability density of finding an electron at a given point in space. 
# For hydrogen-like atoms, the charge density can be calculated from the wavefunctions of the bound states. 
# Each bound state is characterized by its energy and angular momentum quantum number, and the wavefunction can be obtained by solving the Schrödinger equation. 
# The charge density is proportional to the square of the wavefunction, integrated over all occupied states. 
# The degeneracy of each state, determined by its angular momentum quantum number, must be considered when populating the available orbitals. 
# The total charge density is the sum of the contributions from all occupied states.

def calculate_charge_density(bound_states, r_grid, Z):
    '''Input
    bound_states: bound states found using the find_bound_states function; a list of tuples
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    charge_density: the calculated charge density coming from the bound states; 1D array of float
    '''
    # Sort the bound states using the sort_states function
    sorted_states = sort_states(bound_states)
    
    # Initialize the charge density array
    charge_density = np.zeros_like(r_grid)
    
    # Total number of electrons to populate
    total_electrons = Z
    
    # Populate the available orbitals with the sorted states
    for l, energy in sorted_states:
        # Calculate the degeneracy of the state
        degeneracy = 2 * (2 * l + 1)
        
        # Determine the number of electrons that can occupy this state
        electrons_in_state = min(degeneracy, total_electrons)
        
        # If there are no more electrons to place, break the loop
        if electrons_in_state <= 0:
            break
        
        # Calculate the wavefunction for this state using the compute_Schrod function
        ur_norm = compute_Schrod(energy, r_grid, l)
        
        # Calculate the contribution to the charge density
        charge_density += electrons_in_state * ur_norm**2
        
        # Decrease the number of electrons left to place
        total_electrons -= electrons_in_state
    
    return charge_density


# Background: In quantum mechanics, the Hartree potential is used to account for the electron-electron interaction in a many-electron system. 
# It is derived from the charge density of the system and is a solution to the Poisson equation. 
# The Poisson equation for the Hartree potential V_H(r) is given by ∇²V_H(r) = -8πρ(r), where ρ(r) is the charge density. 
# In spherical coordinates, this equation can be solved using a numerical method such as the Numerov algorithm. 
# The solution provides the Hartree potential, which is then used to modify the effective potential in the Schrödinger equation. 
# The Hartree potential is often expressed in terms of a function U(r) = V_H(r)r, which simplifies the numerical solution.

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
    # Constants
    e = 1.60217662e-19  # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

    # Convert charge density to SI units (C/m^3)
    charge_density_SI = charge_density * e / (4 * np.pi * epsilon_0)

    # Calculate f(r) for the Poisson equation: u''(r) = -8πρ(r)u(r)
    f_r = -8 * np.pi * charge_density_SI

    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step)

    # Calculate Hartree potential U(r) = V_H(r) * r
    HartreeU = u_r / r_grid

    return HartreeU


# Background: In quantum mechanics, the Schrödinger equation for a hydrogen-like atom can be modified to include the Hartree potential, 
# which accounts for electron-electron interactions. The Hartree potential is derived from the charge density and is included in the 
# effective potential of the system. The radial part of the Schrödinger equation can be expressed in the form u''(r) = f(r)u(r), 
# where f(r) includes contributions from the Coulomb potential, the centrifugal term due to angular momentum, and the Hartree potential. 
# The function f(r) is crucial for solving the differential equation using numerical methods like the Numerov method. 
# The Hartree potential modifies the effective potential, and thus the function f(r) must be recalculated to include this term.

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
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J·s
    m_e = 9.10938356e-31  # Electron mass in kg
    e = 1.60217662e-19    # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

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
            # Effective potential term including Hartree potential
            V_eff = (-Ze2_over_4pi_epsilon0 / r + 
                     l * (l + 1) * hbar2_over_2m / r**2 + 
                     hartreeU[i] / r)
            # f(r) = (2m/ħ²) * (V_eff - E)
            f_r[i] = (2 * m_e / hbar**2) * (V_eff - energy)

    return f_r


# Background: In quantum mechanics, solving the Schrödinger equation with the Hartree potential involves 
# accounting for electron-electron interactions in a many-electron system. The Hartree potential modifies 
# the effective potential in the Schrödinger equation, which is expressed in the form u''(r) = f(r)u(r). 
# The function f(r) includes contributions from the Coulomb potential, the centrifugal term due to angular 
# momentum, and the Hartree potential. The Numerov method is a numerical technique used to solve such 
# differential equations. After solving for the radial wave function u(r), it is important to normalize 
# the wave function using Simpson's rule to ensure the total probability of finding the electron is 1.

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


    # Calculate f(r) using the f_Schrod_Hartree function
    f_r = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)

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


# Background: In quantum mechanics, the behavior of wavefunctions near the origin is crucial for understanding the properties of quantum systems. 
# For hydrogen-like atoms, the wavefunction u(r) can be extrapolated to r=0 using numerical techniques. 
# One such technique is polynomial fitting, where a polynomial is fitted to the wavefunction values at the first few grid points. 
# This allows for an accurate estimation of the wavefunction at r=0, which is important for boundary condition checks and normalization. 
# The polyfit function in Numpy can be used to fit a polynomial of a specified order to the data points. 
# In this context, we use a 3rd order polynomial to fit the wavefunction values after dividing by r^l, where l is the angular momentum quantum number.

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

    # Compute the normalized wavefunction using the compute_Schrod_Hartree function
    ur_norm = compute_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
    # Divide the wavefunction by r^l
    ur_divided = ur_norm / (r_grid**l)
    
    # Use the first four points for polynomial fitting
    r_points = r_grid[:4]
    u_points = ur_divided[:4]
    
    # Fit a 3rd order polynomial to the points
    poly_coeffs = np.polyfit(r_points, u_points, 3)
    
    # Evaluate the polynomial at r=0 to extrapolate the value
    u0 = np.polyval(poly_coeffs, 0)
    
    return u0



# Background: In quantum mechanics, finding bound states involves identifying energy levels where the wavefunction satisfies specific boundary conditions. 
# For systems with electron-electron interactions, the Hartree potential modifies the effective potential in the Schrödinger equation. 
# The extrapolate_polyfit function is used to estimate the wavefunction at r=0, which is crucial for checking boundary conditions. 
# The brentq method from scipy is a robust root-finding algorithm that can be used to find the zeros of a function within a specified interval. 
# By applying the extrapolate_polyfit function to different energies and using brentq to find where the wavefunction at r=0 is zero, we can identify the bound states.


def find_bound_states_Hartree(r_grid, l, energy_grid, Z, hartreeU):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Z: atomic number; int
    hartreeU: the values of the Hartree term U(r) in the form of U(r)=V_H(r)r, where V_H(r) is the actual Hartree potential term in the Schrodinger equation; a 1d array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    bound_states = []
    max_bound_states = 10
    num_found = 0

    # Iterate over the energy grid to find bound states
    for i in range(len(energy_grid) - 1):
        if num_found >= max_bound_states:
            break

        # Define the function to find the root of
        def shoot_energy(energy):
            return extrapolate_polyfit(energy, r_grid, l, Z, hartreeU)

        # Check if there is a sign change between energy_grid[i] and energy_grid[i+1]
        if shoot_energy(energy_grid[i]) * shoot_energy(energy_grid[i + 1]) < 0:
            # Use brentq to find the root (bound state energy)
            try:
                bound_energy = optimize.brentq(shoot_energy, energy_grid[i], energy_grid[i + 1])
                bound_states.append((l, bound_energy))
                num_found += 1
            except ValueError:
                # If brentq fails, continue to the next interval
                continue

    return bound_states


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('12.12', 3)
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
assert np.allclose(find_bound_states_Hartree(r_grid, 0, energy_grid, Z, hu), target)
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
assert np.allclose(find_bound_states_Hartree(r_grid, 0, energy_grid, Z, hu), target)
target = targets[2]

energy_grid = -1.2/np.arange(1,20,0.2)**2
r_grid = np.linspace(1e-8,100,2000)
Z=28
nmax = 5
bound_states=[]
for l in range(nmax):
    bound_states += find_bound_states(r_grid, l, energy_grid)
charge_density = calculate_charge_density(bound_states,r_grid,Z)
hu = calculate_HartreeU(charge_density, 0.0, 0.5, r_grid[0]-r_grid[1], r_grid, Z)
assert np.allclose(find_bound_states_Hartree(r_grid, 2, energy_grid, Z, hu), target)
