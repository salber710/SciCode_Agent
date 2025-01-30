from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    u, u_prime = y
    u_double_prime = u * (l * (l + 1) / r**2 - 2 / r + 2 * En)
    return [u_prime, u_double_prime]




def SolveSchroedinger(y0, En, l, R):
    def schroedinger_deriv(r, y):
        # Using Schroed_deriv to get the second derivative directly
        return [y[1], Schroed_deriv(y, r, l, En)[1]]

    # Reverse R for integration from large to small r
    R_reversed = R[::-1]
    # Solve the differential equation
    sol = solve_ivp(schroedinger_deriv, [R_reversed[0], R_reversed[-1]], y0, t_eval=R_reversed, method='RK45')
    # Extract the solution and reverse it to match the original R
    u_r = sol.y[0][::-1]

    # Normalize the result using Simpson's rule
    norm_factor = np.trapz(u_r**2, R)
    ur = u_r / np.sqrt(norm_factor)

    return ur



# Background: In quantum mechanics, the wavefunction describes the quantum state of a system. For the radial part of the Schrödinger equation, the wavefunction u(r) is often solved numerically over a range of radii. However, the value of the wavefunction at r=0 is not directly computed due to singularities at the origin. Instead, we can extrapolate this value using the wavefunction values at the first two grid points. This is done by dividing the wavefunction by r^l to account for the behavior of the wavefunction near the origin, where l is the angular momentum quantum number. The extrapolation is typically linear, assuming the wavefunction behaves smoothly near the origin.

def Shoot(En, R, l, y0):
    '''Extrapolate u(0) based on results from SolveSchroedinger function
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    Output 
    f_at_0: Extrapolate u(0), float
    '''
    # Solve the Schrödinger equation to get the wavefunction values
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Get the first two points in the radial grid
    r1, r2 = R[0], R[1]
    u1, u2 = ur[0], ur[1]
    
    # Divide the wavefunction by r^l to account for the angular momentum
    u1_div = u1 / (r1**l)
    u2_div = u2 / (r2**l)
    
    # Linearly extrapolate to find the value at r=0
    # Using the formula for linear extrapolation: f(0) = f(x1) + (f(x2) - f(x1)) * (0 - x1) / (x2 - x1)
    f_at_0 = u1_div + (u2_div - u1_div) * (0 - r1) / (r2 - r1)
    
    return f_at_0


try:
    targets = process_hdf5_to_tuple('52.3', 3)
    target = targets[0]
    assert np.allclose(Shoot( 1.1, np.logspace(1,2.2,10), 3, [0, -1e-5]), target)

    target = targets[1]
    assert np.allclose(Shoot(2.1, np.logspace(-2,2,100), 2, [0, -1e-5]), target)

    target = targets[2]
    assert np.allclose(Shoot(2, np.logspace(-3,3.2,1000), 1, [0, -1e-5]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e