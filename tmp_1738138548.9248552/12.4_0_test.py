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



# Background: In quantum mechanics, the wavefunction of a particle describes the quantum state of a system. 
# For a given potential, the Schrödinger equation can be solved to find the wavefunction. 
# In the context of the radial Schrödinger equation, the wavefunction is often expressed in terms of the radial 
# coordinate r. When solving the equation numerically, especially using methods like the Numerov method, 
# the wavefunction is computed at discrete points on a radial grid. 
# To estimate the behavior of the wavefunction at r=0, where it is not directly computed, 
# we can use linear extrapolation based on the values at the first two grid points. 
# Before extrapolating, it is important to account for the angular momentum quantum number l, 
# which affects the behavior of the wavefunction near the origin. 
# Specifically, the wavefunction should be divided by r^l to remove the singularity introduced by the angular momentum.

def shoot(energy, r_grid, l):
    '''Input 
    energy: a float
    r_grid: the radial grid; a 1D array of float
    l: angular momentum quantum number; an int
    Output 
    f_at_0: float
    '''
    # Compute the wavefunction using the previously defined compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)
    
    # Divide the wavefunction by r^l
    ur_divided = ur_norm / (r_grid**l)
    
    # Perform linear extrapolation to estimate the value at r=0
    # Using the first two points in the radial grid
    r1, r2 = r_grid[0], r_grid[1]
    u1, u2 = ur_divided[0], ur_divided[1]
    
    # Linear extrapolation formula: f(0) = u1 + (0 - r1) * (u2 - u1) / (r2 - r1)
    f_at_0 = u1 + (-r1) * (u2 - u1) / (r2 - r1)
    
    return f_at_0


try:
    targets = process_hdf5_to_tuple('12.4', 3)
    target = targets[0]
    assert np.allclose(shoot(1.1, np.linspace(1e-7,20,10), 0), target)

    target = targets[1]
    assert np.allclose(shoot(1.1, np.linspace(1e-7,20,10), 1), target)

    target = targets[2]
    assert np.allclose(shoot(1.1, np.linspace(1e-7,50,10), 2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e