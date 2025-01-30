from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from scipy import integrate
from scipy import optimize
import numpy as np


def f_Schrod(energy, l, r_grid):
    # Constants
    hbar = 1.0545718e-34  # Reduced Planck's constant in J*s
    m = 9.10938356e-31    # Mass of electron in kg
    e = 1.602176634e-19   # Elementary charge in C
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    Z = 1  # Atomic number for hydrogen

    # Precompute constants
    factor = 2 * m / hbar**2
    V_prefactor = -Z * e**2 / (4 * np.pi * epsilon_0)

    # Calculate f(r) using numpy operations for efficiency
    V_r = np.divide(V_prefactor, r_grid, out=np.full_like(r_grid, -np.inf), where=r_grid!=0)
    centrifugal_term = np.divide(l * (l + 1), r_grid**2, out=np.full_like(r_grid, np.inf), where=r_grid!=0)
    f_r = factor * (V_r - energy) + centrifugal_term

    # Handle special case at r = 0
    if r_grid[0] == 0:
        f_r[0] = factor * (-energy) + l * (l + 1) / r_grid[1]**2  # Use small r value for approximation

    return f_r



def Numerov(f_in, u_at_0, up_at_0, step):
    n = len(f_in)
    u = np.zeros(n)
    u[0] = u_at_0
    if n > 1:
        u[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * f_in[0] * u_at_0

    for i in range(1, n - 1):
        u[i + 1] = (2 * u[i] - u[i - 1] + (step**2 / 12) * (10 * f_in[i] * u[i] + f_in[i + 1] * u[i] + f_in[i - 1] * u[i - 1])) / (1 - (step**2 / 12) * f_in[i + 1])

    return u



def compute_Schrod(energy, r_grid, l):
    f_r = f_Schrod(energy, l, r_grid)
    step = r_grid[0] - r_grid[1]
    u_r = Numerov(f_r, 0, -1e-7, step)
    
    # Reverse u_r for normalization from the largest radius
    u_r_reversed = u_r[::-1]
    r_grid_reversed = r_grid[::-1]
    
    # Normalize using Simpson's rule
    integral = np.trapz(u_r_reversed**2, r_grid_reversed)
    ur_norm = u_r / np.sqrt(integral)
    
    return ur_norm


def shoot(energy, r_grid, l):


    # Compute the wavefunction at the grid points
    wavefunction = compute_Schrod(energy, r_grid, l)
    
    # Normalize the wavefunction by dividing by r^l, handling r=0 case
    normalized_wavefunction = np.where(r_grid != 0, wavefunction / r_grid**l, 0)
    
    # Use a hyperbolic tangent model to extrapolate to r=0 using the first two points
    # Fit a function of the form a * tanh(b * r) + c
    # Approximate b as the inverse of the average of the first two r values
    b_approx = 1 / np.mean(r_grid[:2])
    a_approx = (normalized_wavefunction[1] - normalized_wavefunction[0]) / (np.tanh(b_approx * r_grid[1]) - np.tanh(b_approx * r_grid[0]))
    c_approx = normalized_wavefunction[0] - a_approx * np.tanh(b_approx * r_grid[0])
    
    # The extrapolated value at r=0 is the constant term c
    f_at_0 = c_approx
    
    return f_at_0



# Background: In quantum mechanics, bound states refer to the discrete energy levels of a particle confined to a potential well, such as an electron in an atom. The Schrödinger equation describes these states, and finding them involves solving the equation for specific energy values where the wavefunction satisfies boundary conditions. The shooting method is a numerical technique used to find these energy levels by adjusting the energy until the wavefunction behaves correctly at the boundaries. The `brentq` method from SciPy is a root-finding algorithm that can be used to find the energy values where the wavefunction satisfies the boundary conditions, indicating a bound state.


def find_bound_states(r_grid, l, energy_grid):
    '''Input
    r_grid: a 1D array of float
    l: angular momentum quantum number; int
    energy_grid: energy grid used for search; a 1D array of float
    Output
    bound_states: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    
    bound_states = []
    
    # Define a function that uses the shoot function to find the value at r=0
    def shoot_function(energy):
        return shoot(energy, r_grid, l)
    
    # Iterate over the energy grid to find bound states
    for i in range(len(energy_grid) - 1):
        E1 = energy_grid[i]
        E2 = energy_grid[i + 1]
        
        # Check if there is a sign change in the shoot function between E1 and E2
        if shoot_function(E1) * shoot_function(E2) < 0:
            # Use brentq to find the root (energy level) where the sign change occurs
            bound_energy = optimize.brentq(shoot_function, E1, E2)
            bound_states.append((l, bound_energy))
            
            # Stop if we have found the maximum number of bound states
            if len(bound_states) >= 10:
                break
    
    return bound_states


try:
    targets = process_hdf5_to_tuple('12.5', 3)
    target = targets[0]
    assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),2, -1.2/np.arange(1,20,0.2)**2), target)

    target = targets[1]
    assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),3,-1.2/np.arange(1,20,0.2)**2), target)

    target = targets[2]
    assert np.allclose(find_bound_states(np.linspace(1e-8,100,2000),0,-1.2/np.arange(1,20,0.2)**2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e