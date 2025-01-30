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



# Background: The Schrödinger equation for a hydrogen-like atom can be solved numerically using the Numerov method, which is a stable and efficient algorithm for solving second-order linear differential equations. The function f_Schrod computes the effective potential term f(r) in the radial Schrödinger equation. The Numerov method then uses this f(r) to compute the radial wavefunction u(r). To ensure the physical validity of the wavefunction, it must be normalized. Simpson's rule is a numerical integration technique that can be used to compute the integral of the square of the wavefunction over the radial grid, which is necessary for normalization.


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
    
    # Initial conditions for Numerov method
    u_at_0 = 0
    up_at_0 = -1e-7
    step = r_grid[1] - r_grid[0]  # Step size for Numerov method
    
    # Solve for u(r) using the Numerov method
    u_r = Numerov(f_r, u_at_0, up_at_0, step)
    
    # Normalize the wavefunction using Simpson's rule
    # Calculate the integral of |u(r)|^2 over the radial grid
    integral = integrate.simpson(u_r**2, r_grid)
    
    # Normalize u(r)
    ur_norm = u_r / np.sqrt(integral)
    
    return ur_norm


try:
    targets = process_hdf5_to_tuple('12.3', 3)
    target = targets[0]
    assert np.allclose(compute_Schrod(1, np.linspace(1e-5,10,20), 1), target)

    target = targets[1]
    assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,10), 2), target)

    target = targets[2]
    assert np.allclose(compute_Schrod(1, np.linspace(1e-5,20,20), 3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e