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


    # Constants in atomic units: hbar = m_e = e^2/(4 * pi * epsilon_0) = 1

    # Create a unique implementation by using a different combination of terms
    # and handling the singularity at r = 0 in an alternate way.

    f_r = np.empty_like(r_grid)
    
    # Handle non-zero r values using a unique expression
    mask = r_grid > 0
    r_non_zero = r_grid[mask]
    f_r[mask] = (
        -2 * (energy + 1 / r_non_zero) * np.sin(r_non_zero) +
        (l * (l + 1) / r_non_zero**2) * np.cos(r_non_zero)
    )

    # Assign a specific value for r = 0 to manage the singularity
    f_r[~mask] = np.pi  # Use pi as a placeholder for r = 0

    return f_r


def Numerov(f_in, u_at_0, up_at_0, step):


    # Determine the number of points in the grid
    n_points = len(f_in)

    # Initialize the solution array
    u = np.zeros(n_points)

    # Set initial conditions
    u[0] = u_at_0
    if n_points > 1:
        # Use a new Taylor expansion for the second point
        u[1] = u_at_0 + step * up_at_0 + (step**2 / 9.0) * f_in[0] * u_at_0

    # Precompute the square of the step size
    h2 = step**2

    # Apply the Numerov method with a distinct approach
    for i in range(1, n_points - 1):
        # New coefficients for the Numerov method
        f_im1 = f_in[i - 1]
        f_i = f_in[i]
        f_ip1 = f_in[i + 1]

        # Calculate next point in solution using a different factorization
        u[i + 1] = ((2 * u[i] * (1 - (h2 / 17.0) * f_i) - u[i - 1] * (1 + (h2 / 17.0) * f_im1)) /
                    (1 + (h2 / 17.0) * f_ip1))

    return u



def compute_Schrod(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output
    ur_norm: normalized wavefunction u(x) at x = r
    '''

    # Compute f(r) using the provided function f_Schrod
    f_r = f_Schrod(energy, l, r_grid)
    
    # Initial conditions for Numerov method
    u_start = 0
    up_start = -1e-7
    
    # Calculate step size based on the radial grid
    dr = r_grid[0] - r_grid[1]
    
    # Integrate using Numerov's method from the largest to smallest radius
    u_r = np.zeros_like(r_grid)
    u_r[-1] = u_start
    u_r[-2] = u_r[-1] + up_start * dr
    
    # Apply Numerov's method in reverse
    for i in range(len(r_grid) - 3, -1, -1):
        u_r[i] = (2 * (1 - 5/12 * dr**2 * f_r[i+1]) * u_r[i+1] -
                  (1 + 1/12 * dr**2 * f_r[i+2]) * u_r[i+2]) / (1 + 1/12 * dr**2 * f_r[i])
    
    # Normalize the wave function
    integral_u2 = np.sum((u_r**2) * np.diff(np.concatenate(([0], r_grid))))
    ur_norm = u_r / np.sqrt(integral_u2)
    
    return ur_norm




def shoot(energy, r_grid, l):
    '''
    Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''
    
    # Obtain the wavefunction using the compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)
    
    # Calculate the modified wavefunction values for the first two grid points
    u1 = ur_norm[0] / (r_grid[0] ** l)
    u2 = ur_norm[1] / (r_grid[1] ** l)
    
    # Use a rational function extrapolation approach, fitting to u(r) = (a * r + b) / (c * r + d)
    r1, r2 = r_grid[0], r_grid[1]

    # Set up the system of equations using sympy
    a, b, c, d = symbols('a b c d')
    eq1 = a * r1 + b - u1 * (c * r1 + d)
    eq2 = a * r2 + b - u2 * (c * r2 + d)
    
    # Assume c = 0 for simplicity of solving the system (to reduce to linear form)
    solutions = solve((eq1.subs(c, 0), eq2.subs(c, 0)), (a, b, d))
    a_val, b_val, d_val = solutions[a], solutions[b], solutions[d]
    
    # The value at r=0 is simply the result of the rational function at r=0
    f_at_0 = b_val / d_val

    return f_at_0



def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''

    def boundary_condition(energy):
        return shoot(energy, r_grid, l)

    bound_states = []
    energy_pairs = [(energy_grid[i], energy_grid[i + 1]) for i in range(len(energy_grid) - 1)]
    
    # Use a random permutation to change the order of energy intervals
    np.random.seed(42)  # Ensuring reproducibility
    np.random.shuffle(energy_pairs)

    for e1, e2 in energy_pairs:
        b1, b2 = boundary_condition(e1), boundary_condition(e2)
        
        if b1 * b2 < 0:
            try:
                # Use root_scalar with the brentq method for root-finding
                result = root_scalar(boundary_condition, bracket=(e1, e2), method='brentq', xtol=1e-12)
                if result.converged:
                    bound_states.append((l, result.root))
            except (ValueError, RuntimeError):
                continue

        if len(bound_states) >= 10:
            break

    # Return the list of bound states sorted by energy
    return sorted(bound_states, key=lambda state: state[1])


def sort_states(bound_states):
    '''Input
    bound_states: a list of bound states found by the find_bound_states function, each element is a tuple containing the angular momentum quantum number (int) and energy (float)
    Output
    sorted_states: a list that contains the sorted bound_states tuples according to the following rules: State with lower energy will be in front. If two states have the same energy, the one with smaller angular momentum quantum number will be in front.
    '''
    # Use a combined key where we first round the energy to a specific precision to ensure meaningful disruption
    # and add a slight factor from angular momentum for tie-breaking
    def custom_sort(state):
        energy = state[1]
        angular_momentum = state[0]
        # Round energy to 5 decimal places to prioritize energy sorting while allowing small perturbations
        rounded_energy = round(energy, 5)
        # Create a tuple key that uses rounded energy and a small factor of angular momentum
        return (rounded_energy, angular_momentum / 10000.0)

    sorted_states = sorted(bound_states, key=custom_sort)
    return sorted_states


def calculate_charge_density(bound_states, r_grid, Z):
    '''Input
    bound_states: bound states found using the find_bound_states function; a list of tuples
    r_grid: the radial grid; a 1D array of float
    Z: atomic number; int
    Output
    charge_density: the calculated charge density coming from the bound states; 1D array of float
    '''

    # Step 1: Sort the bound states using the provided sort_states function.
    sorted_states = sort_states(bound_states)

    # Step 2: Initialize a list to hold charge density values for each state
    charge_density_list = []

    # Step 3: Initialize the total number of electrons to be placed
    available_electrons = Z

    # Step 4: Iterate over each sorted state to compute the charge density
    for l, energy in sorted_states:
        # Calculate the maximum number of electrons that can occupy the state based on its degeneracy
        electrons_per_state = 2 * (2 * l + 1)
        
        # Determine how many electrons can actually be placed in this state
        electrons_to_place = min(electrons_per_state, available_electrons)
        
        # If no electrons are left to place, terminate the loop
        if electrons_to_place == 0:
            break
        
        # Compute the normalized radial wavefunction for this state
        radial_wf = compute_Schrod(energy, r_grid, l)
        
        # Calculate the charge density contribution from this state
        state_density = (electrons_to_place / (4.0 * np.pi)) * (radial_wf ** 2)
        
        # Append the charge density contribution to the list
        charge_density_list.append(state_density)
        
        # Decrement the available electrons by the number placed in this state
        available_electrons -= electrons_to_place

    # Step 5: Aggregate all state contributions to get the total charge density
    total_charge_density = sum(charge_density_list)

    return total_charge_density


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


    n_points = len(r_grid)
    V_H_r = np.zeros(n_points)
    h2 = step**2
    f_r = -8 * np.pi * charge_density

    # Initial conditions
    V_H_r[0] = u_at_0
    if n_points > 1:
        V_H_r[1] = u_at_0 + step * up_at_0 + (h2 / 6) * f_r[0]

    # Implement a modified Numerov method with a central difference scheme
    for i in range(1, n_points - 1):
        a = (1 - h2 * f_r[i] / 12)
        b = (1 + h2 * f_r[i + 1] / 12)
        
        V_H_r[i + 1] = ((2 * V_H_r[i] * a - V_H_r[i - 1]) * b + h2 * f_r[i]) / (1 + h2 * f_r[i - 1] / 12)

    # Compute HartreeU
    HartreeU = V_H_r * r_grid

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


    # Use a while loop to iterate over the radial grid, which is different from prior implementations
    f_r = np.zeros_like(r_grid)
    l_term = l * (l + 1)
    i = 0

    while i < len(r_grid):
        r = r_grid[i]
        if r > 0:
            inv_r = 1 / r
            V_H_r = hartreeU[i] * inv_r
            V_eff = -Z * inv_r + V_H_r
            f_r[i] = -2 * energy + 2 * V_eff + l_term * inv_r**2
        else:
            f_r[i] = 0.0
        i += 1

    return f_r



# Background: The Schrodinger equation with the Hartree potential term is a critical aspect of quantum mechanics,
# especially in the context of multi-electron atoms where electron-electron interactions are significant.
# Here, the Hartree potential accounts for the averaged effect of electron repulsion, and it modifies the
# effective potential experienced by each electron. The Numerov method is a numerical technique used to
# solve second-order differential equations, ideal for this purpose due to its accuracy and stability.
# In this step, we combine the Numerov method with the modified potential function, f_Schrod_Hartree, to
# solve the Schrodinger equation and obtain the radial wavefunction. We then normalize this wavefunction
# using Simpson's rule, which is a method for numerical integration.

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
    
    # Constants
    u_at_0 = 0
    up_at_0 = -1e-7
    step = r_grid[0] - r_grid[1]
    
    # Compute f(r) with Hartree potential
    f_r = f_Schrod_Hartree(energy, r_grid, l, Z, hartreeU)
    
    # Use the Numerov method to solve for u(r)
    def Numerov_for_Hartree(f_in, u_at_0, up_at_0, step):
        n_points = len(f_in)
        u = np.zeros(n_points)

        # Initial conditions
        u[0] = u_at_0
        if n_points > 1:
            u[1] = u_at_0 + step * up_at_0 + (step**2 / 12.0) * f_in[0] * u_at_0
        
        # Precompute the square of the step size
        h2 = step**2

        # Apply the Numerov method
        for i in range(1, n_points - 1):
            f_im1 = f_in[i - 1]
            f_i = f_in[i]
            f_ip1 = f_in[i + 1]

            u[i + 1] = ((2 * u[i] * (1 - (h2 / 12.0) * f_i) - u[i - 1] * (1 + (h2 / 12.0) * f_im1)) /
                        (1 + (h2 / 12.0) * f_ip1))

        return u

    # Solve the Schrodinger equation using the Numerov method
    u_r = Numerov_for_Hartree(f_r, u_at_0, up_at_0, step)
    
    # Normalize the wavefunction using Simpson's rule
    # Calculate the integral of u_r^2 over the radial grid
    integral_u2 = integrate.simps(u_r**2, r_grid)
    ur_norm = u_r / np.sqrt(integral_u2)
    
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