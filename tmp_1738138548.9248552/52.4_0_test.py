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


def Shoot(En, R, l, y0):
    # Solve the Schrödinger equation to get the wavefunction values
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Extract the first two points in the radial grid and their corresponding wavefunction values
    r1, r2 = R[0], R[1]
    u1, u2 = ur[0], ur[1]
    
    # Normalize the wavefunction by r^l, ensuring no division by zero
    u1_normalized = u1 / (r1 ** l) if r1 > 0 else 0
    u2_normalized = u2 / (r2 ** l) if r2 > 0 else 0
    
    # Use a sinusoidal extrapolation to estimate u(0)
    # Assuming a sinusoidal fit u = A * sin(B * r + C) + D, we solve for A, B, C, D using two points
    if r1 > 0 and r2 > 0:

        
        def equations(vars):
            A, B, C, D = vars
            eq1 = A * np.sin(B * r1 + C) + D - u1_normalized
            eq2 = A * np.sin(B * r2 + C) + D - u2_normalized
            return [eq1, eq2]
        
        A, B, C, D = fsolve(equations, (1, 1, 1, 1))
    else:
        # Fallback to a simpler estimation if one of the radii is zero
        A, B, C, D = 0, 1, 0, u1_normalized if r1 > 0 else u2_normalized
    
    # Extrapolate to find the wavefunction at r=0
    f_at_0 = A * np.sin(B * 0 + C) + D
    
    return f_at_0



# Background: In quantum mechanics, bound states are solutions to the Schrödinger equation where the particle is confined to a finite region of space, typically due to a potential well. These states have discrete energy levels, known as eigenvalues. The shooting method is a numerical technique used to find these eigenvalues by iteratively adjusting the energy until the boundary conditions are satisfied. In this context, we use the Shoot function to evaluate the wavefunction at r=0 for different energies. By searching through a range of energies (Esearch), we can identify energies that correspond to bound states by looking for sign changes in the wavefunction at r=0, which indicate a crossing through zero, a characteristic of eigenvalues.

def FindBoundStates(y0, R, l, nmax, Esearch):
    '''Input
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    R: an 1D array of (logspace) of radius; each element is a float
    l: angular momentum quantum number, int
    nmax: maximum number of bounds states wanted, int
    Esearch: energy mesh used for search, an 1D array of float
    Output
    Ebnd: a list, each element is a tuple containing the angular momentum quantum number (int) and energy (float) of all bound states found
    '''
    
    Ebnd = []
    previous_sign = None
    
    for En in Esearch:
        # Use the Shoot function to get the wavefunction value at r=0
        f_at_0 = Shoot(En, R, l, y0)
        
        # Determine the sign of the wavefunction at r=0
        current_sign = np.sign(f_at_0)
        
        # Check for a sign change, indicating a bound state
        if previous_sign is not None and current_sign != previous_sign:
            # A sign change indicates a crossing through zero, suggesting a bound state
            Ebnd.append((l, En))
            
            # Stop if we have found the maximum number of bound states
            if len(Ebnd) >= nmax:
                break
        
        # Update the previous sign
        previous_sign = current_sign
    
    return Ebnd


try:
    targets = process_hdf5_to_tuple('52.4', 3)
    target = targets[0]
    y0 = [0, -1e-5]
    Esearch = -1.2/np.arange(1,20,0.2)**2
    R = np.logspace(-6,2.2,500)
    nmax=7
    Bnd=[]
    for l in range(nmax-1):
        Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
    assert np.allclose(Bnd, target)

    target = targets[1]
    Esearch = -0.9/np.arange(1,20,0.2)**2
    R = np.logspace(-8,2.2,1000)
    nmax=5
    Bnd=[]
    for l in range(nmax-1):
        Bnd += FindBoundStates(y0, R,l,nmax-l,Esearch)
    assert np.allclose(Bnd, target)

    target = targets[2]
    Esearch = -1.2/np.arange(1,20,0.2)**2
    R = np.logspace(-6,2.2,500)
    nmax=7
    Bnd=[]
    for l in range(nmax-1):
        Bnd += FindBoundStates(y0,R,l,nmax-l,Esearch)
    assert np.isclose(Bnd[0], (0,-1)).any() == target.any()

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e