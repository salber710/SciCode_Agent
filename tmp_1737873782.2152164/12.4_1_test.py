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

    # Compute the normalized wavefunction using the compute_Schrod function
    ur_norm = compute_Schrod(energy, r_grid, l)
    
    # Extrapolate the value at r=0 using the wavefunctions at the first two grid points
    # Divide the wavefunction by r^l before extrapolation
    r1, r2 = r_grid[0], r_grid[1]
    u1 = ur_norm[0] / (r1 ** l)
    u2 = ur_norm[1] / (r2 ** l)
    
    # Linear extrapolation to find the wavefunction value at r=0
    f_at_0 = u1 + (u1 - u2) * (0 - r1) / (r2 - r1)

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