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
    
    # Define the shooting function to find roots
    def shooting_function(energy):
        return shoot(energy, r_grid, l)
    
    # Loop through the energy grid and find roots
    for i in range(len(energy_grid) - 1):
        energy1, energy2 = energy_grid[i], energy_grid[i + 1]
        
        # Check if the function changes sign between energy1 and energy2
        if shooting_function(energy1) * shooting_function(energy2) < 0:
            # Use brentq to find a root in the interval [energy1, energy2]
            try:
                energy_root = optimize.brentq(shooting_function, energy1, energy2)
                bound_states.append((l, energy_root))
                
                # Stop if maximum number of bound states is reached
                if len(bound_states) >= 10:
                    break
                    
            except ValueError:
                # If brentq fails, continue to the next interval
                continue
    
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