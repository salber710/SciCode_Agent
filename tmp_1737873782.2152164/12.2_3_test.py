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

    n = len(f_in)
    u = np.zeros(n)

    # Initial conditions
    u[0] = u_at_0
    u[1] = u_at_0 + step * up_at_0 + (step**2 / 2) * f_in[0] * u_at_0

    # Numerov method iteration
    step_squared = step**2

    for i in range(1, n-1):
        u[i+1] = (2 * (1 - (5/12) * step_squared * f_in[i]) * u[i] - 
                  (1 + (1/12) * step_squared * f_in[i-1]) * u[i-1]) / \
                 (1 + (1/12) * step_squared * f_in[i+1])

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