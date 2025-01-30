from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np

def f_Schrod(energy, l, r_grid):
    '''Input 
    energy: a float
    l: angular momentum quantum number; an int
    r_grid: the radial grid; a 1D array of float
    Output
    f_r: a 1D array of float 
    '''
    # Constants
    h_bar = 1.0545718e-34  # Planck's constant over 2π in J·s
    m_e = 9.10938356e-31   # Mass of an electron in kg
    e_charge = 1.60217662e-19  # Elementary charge in coulombs
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

    # Reduced mass for hydrogen-like atom (assuming Z=1 and nucleus is a proton)
    mu = m_e  # For hydrogen, the reduced mass is approximately the electron mass

    # Pre-computed constant factor in the equation
    factor = -h_bar**2 / (2 * mu)

    # Initialize the output array
    f_r = np.zeros_like(r_grid)

    # Calculate f(r) for each radius in r_grid
    for i, r in enumerate(r_grid):
        if r != 0:  # Avoid division by zero
            # Coulomb potential term
            V_r = -e_charge**2 / (4 * np.pi * epsilon_0 * r)
        
            # Centrifugal term due to angular momentum
            centrifugal_term = (l * (l + 1) * h_bar**2) / (2 * mu * r**2)

            # Calculate f(r) using the relation from the modified Schrödinger equation
            f_r[i] = (2 * mu / h_bar**2) * (V_r - energy + centrifugal_term)
        else:
            # Handle the r=0 case separately if needed
            f_r[i] = np.inf  # or some large value to indicate singularity

    return f_r




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

    # Initialize the array to hold the solution u(r)
    u = np.zeros(n_points)

    # Set initial conditions
    u[0] = u_at_0
    # Use Taylor expansion to estimate the next point based on the derivative
    u[1] = u_at_0 + step * up_at_0 + (step**2 / 2) * f_in[0] * u_at_0

    # Numerov iteration to solve for u(r)
    for i in range(1, n_points - 1):
        u[i+1] = (2 * (1 - (5 * step**2 / 12) * f_in[i]) * u[i] - 
                  (1 + (step**2 / 12) * f_in[i-1]) * u[i-1]) / (1 + (step**2 / 12) * f_in[i+1])

    return u



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

    # Determine the step size for the Numerov method
    step_size = r_grid[0] - r_grid[1]

    # Solve for u(r) using the Numerov method
    u = Numerov(f_r, 0.0, -1e-7, step_size)

    # Normalize u(r) using Simpson's rule
    # Simpson's rule requires an odd number of points, ensure it is suitable
    if len(u) % 2 == 0:
        u = u[:-1]
        r_grid = r_grid[:-1]

    # Calculate the integral of u(r)^2
    norm = np.sqrt(integrate.simpson(u**2, r_grid))

    # Normalize u(r)
    ur_norm = u / norm

    return ur_norm



def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''

    # Compute the normalized wavefunction u(r) using the compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)

    # Extract the first two points from the normalized wavefunction and r_grid
    u1, u2 = ur_norm[:2]
    r1, r2 = r_grid[:2]

    # Divide the wavefunction values by r^l
    u1_div = u1 / (r1**l)
    u2_div = u2 / (r2**l)

    # Linearly extrapolate to find the value at r = 0
    f_at_0 = u1_div - (u2_div - u1_div) / (r2 - r1) * r1

    return f_at_0


def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''

    bound_states = []

    # Define the function whose root we want to find using the shoot function
    def shoot_function(energy):
        return shoot(energy, r_grid, l)

    # Maximum number of bound states to search for
    max_bound_states = 10

    # Iterate over the energy grid and find roots (bound states)
    for i in range(len(energy_grid) - 1):
        energy_start, energy_end = energy_grid[i], energy_grid[i+1]

        # Check if there is a sign change between the shoot function values at energy_start and energy_end
        if shoot_function(energy_start) * shoot_function(energy_end) < 0:
            try:
                # Use the brentq method to find the root (energy) where the shoot function is zero
                bound_energy = optimize.brentq(shoot_function, energy_start, energy_end)
                bound_states.append((l, bound_energy))

                # Stop if we've found the maximum number of bound states
                if len(bound_states) >= max_bound_states:
                    break

            except ValueError:
                # If brentq fails to find a root, continue to the next interval
                continue

    return bound_states


def sort_states(bound_states):
    '''Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''

    # Sort the bound states primarily by energy and secondarily by angular momentum quantum number
    sorted_states = sorted(bound_states, key=lambda state: (state[1], state[0] / 10000.0))

    return sorted_states


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

    # Define the maximum number of electrons that can occupy each state based on angular momentum quantum number
    # l = 0: s orbital, can hold 2 electrons
    # l = 1: p orbital, can hold 6 electrons
    # l = 2: d orbital, can hold 10 electrons
    # l = 3: f orbital, can hold 14 electrons
    max_electrons_per_l = {0: 2, 1: 6, 2: 10, 3: 14}

    # Total number of electrons available (equal to atomic number for neutral atom)
    total_electrons = Z

    # Populate available orbitals with the sorted states
    for l, energy in sorted_states:
        if total_electrons <= 0:
            break

        # Determine the number of electrons that can occupy this state
        electrons_in_state = min(total_electrons, max_electrons_per_l.get(l, 0))
        total_electrons -= electrons_in_state

        # Calculate the wavefunction for this state using the compute_Schrod function
        ur_norm = compute_Schrod(energy, r_grid, l)

        # Calculate the charge density contribution from this state
        density_contribution = electrons_in_state * ur_norm**2

        # Add the contribution to the total charge density
        charge_density += density_contribution

    return charge_density


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

    # Calculate the right-hand side of the Poisson equation
    # This is -8 * pi * charge_density
    rhs = -8 * np.pi * charge_density

    # Use Numerov method to solve for V_H(r)
    # Initialize the array to hold the solution V_H(r)
    V_H = np.zeros_like(r_grid)

    # Set initial conditions for Numerov
    V_H[0] = u_at_0
    V_H[1] = u_at_0 + step * up_at_0 + (step**2 / 2) * rhs[0] * u_at_0

    # Numerov iteration to solve for V_H(r)
    for i in range(1, len(r_grid) - 1):
        V_H[i+1] = (2 * (1 - (5 * step**2 / 12) * rhs[i]) * V_H[i] - 
                      (1 + (step**2 / 12) * rhs[i-1]) * V_H[i-1]) / (1 + (step**2 / 12) * rhs[i+1])

    # Calculate the HartreeU term U(r) = V_H(r) * r
    HartreeU = V_H * r_grid

    return HartreeU


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
    h_bar = 1.0545718e-34  # Planck's constant over 2π in J·s
    m_e = 9.10938356e-31   # Mass of an electron in kg
    e_charge = 1.60217662e-19  # Elementary charge in coulombs
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

    # Reduced mass for hydrogen-like atom (assuming Z=1 and nucleus is a proton)
    mu = m_e  # For hydrogen, the reduced mass is approximately the electron mass

    # Pre-computed constant factor in the equation
    factor = -h_bar**2 / (2 * mu)

    # Initialize the output array
    f_r = np.zeros_like(r_grid)

    # Calculate f(r) for each radius in r_grid
    for i, r in enumerate(r_grid):
        if r != 0:  # Avoid division by zero
            # Coulomb potential term
            V_r = -Z * e_charge**2 / (4 * np.pi * epsilon_0 * r)
        
            # Centrifugal term due to angular momentum
            centrifugal_term = (l * (l + 1) * h_bar**2) / (2 * mu * r**2)

            # Hartree potential term, V_H(r) = U(r) / r
            V_H_r = hartreeU[i] / r

            # Calculate f(r) using the relation from the modified Schrödinger equation with Hartree potential
            f_r[i] = (2 * mu / h_bar**2) * (V_r + V_H_r - energy + centrifugal_term)
        else:
            # Handle the r=0 case separately if needed
            f_r[i] = np.inf  # or some large value to indicate singularity

    return f_r



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

    # Calculate f(r) using the provided f_Schrod_Hartree function
    f_r = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)

    # Determine the step size for the Numerov method
    step_size = r_grid[0] - r_grid[1]

    # Solve for u(r) using the Numerov method
    u = Numerov(f_r, 0.0, -1e-7, step_size)

    # Normalize u(r) using Simpson's rule
    # Simpson's rule requires an odd number of points, ensure it is suitable
    if len(u) % 2 == 0:
        u = u[:-1]
        r_grid = r_grid[:-1]

    # Calculate the integral of u(r)^2
    norm = np.sqrt(integrate.simpson(u**2, r_grid))

    # Normalize u(r)
    ur_norm = u / norm

    return ur_norm


try:
    targets = process_hdf5_to_tuple('12.10', 3)
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
    assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 2, Z, hu), target)

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
    assert np.allclose(compute_Schrod_Hartree(-0.4, r_grid, 2, Z, hu), target)

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
    assert np.allclose(compute_Schrod_Hartree(-0.5, r_grid, 3, Z, hu), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e