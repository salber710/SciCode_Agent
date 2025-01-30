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

    
    n_points = len(f_in)
    u = np.zeros(n_points)
    
    # Initial conditions
    u[0] = u_at_0
    if n_points > 1:
        # Using an alternate Taylor expansion for the second point
        u[1] = u_at_0 + step * up_at_0 + (step**2 / 4) * f_in[0] * u_at_0
    
    # Precompute the square of the step size
    h2 = step**2
    
    # Apply the Numerov method with distinct coefficients and arrangement
    for i in range(1, n_points - 1):
        alpha = 1 - (h2 / 9) * f_in[i + 1]
        beta = 2 * (1 + (h2 / 6) * f_in[i])
        gamma = 1 - (h2 / 9) * f_in[i - 1]
        
        # Calculate the next point with a different approach
        u[i + 1] = (beta * u[i] - gamma * u[i - 1]) / alpha
    
    return u


try:
    targets = process_hdf5_to_tuple('12.2', 3)
    target = targets[0]
    assert np.allclose(Numerov(f_Schrod(1,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)

    target = targets[1]
    assert np.allclose(Numerov(f_Schrod(1,2, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)

    target = targets[2]
    assert np.allclose(Numerov(f_Schrod(2,3, np.linspace(1e-5,10,20)), 0.0, -1e-10, np.linspace(1e-5,10,20)[0]-np.linspace(1e-5,10,20)[1]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e