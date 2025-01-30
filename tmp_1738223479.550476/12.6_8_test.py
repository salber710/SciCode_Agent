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
    # Implement a custom sort using a helper function that modifies the tuple for sorting
    def modified_state(state):
        angular_momentum = state[0]
        energy = state[1]
        # Use a large factor to ensure primary sorting by energy and secondary sorting by angular momentum
        return (round(energy * 10000), angular_momentum)

    # Sort using the modified state key
    sorted_states = sorted(bound_states, key=modified_state)
    
    return sorted_states


try:
    targets = process_hdf5_to_tuple('12.6', 3)
    target = targets[0]
    bound_states=[]
    for l in range(6):
        bound_states += find_bound_states(np.linspace(1e-8,100,2000),l,-1.2/np.arange(1,20,0.2)**2)
    assert np.allclose(sort_states(bound_states), target)

    target = targets[1]
    bound_states=[]
    for l in range(3):
        bound_states += find_bound_states(np.linspace(1e-8,100,2000),l,-1.2/np.arange(1,20,0.2)**2)
    assert np.allclose(sort_states(bound_states), target)

    target = targets[2]
    bound_states=[]
    for l in range(1):
        bound_states += find_bound_states(np.linspace(1e-8,100,2000),l,-1.2/np.arange(1,20,0.2)**2)
    assert np.allclose(sort_states(bound_states), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e