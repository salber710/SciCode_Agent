from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize

def Schroed_deriv(y, r, l, En):
    '''Calculate the derivative of y given r, l and En
    Input 
    y=[u,u'], a list of float where u is the wave function at r, u' is the first derivative of u at r
    r: radius, float
    l: angular momentum quantum number, int
    En: energy, float
    Output
    Schroed: dy/dr=[u',u''] , a 1D array of float where u is the wave function at r, u' is the first derivative of u at r, u'' is the second derivative of u at r
    '''
    # Constants
    Z = 1  # Nuclear charge for hydrogen atom
    a0 = 1  # Bohr radius, set to 1 for simplicity as per instructions

    # Unpack y
    u, up = y  # u is the wave function, up is the first derivative of u

    # Calculate the second derivative of u (u'')
    # Based on the radial part of the Schrödinger equation
    upp = (2 * r * up + (2 * En * r**2 - l * (l + 1)) * u) / r**2

    # Return the derivative of y
    return np.array([up, upp])




def SolveSchroedinger(y0, En, l, R):
    '''Integrate the derivative of y within a certain radius
    Input 
    y0: Initial guess for function and derivative, list of floats: [u0, u0']
    En: energy, float
    l:  angular momentum quantum number, int
    R:  an 1d array (linespace) of radius (float)
    Output
    ur: the integration result, float
    '''
    # Integrate using solve_ivp from scipy.integrate
    sol = integrate.solve_ivp(
        fun=lambda r, y: Schroed_deriv(y, r, l, En),
        t_span=(R[0], R[-1]),
        y0=y0,
        t_eval=R,
        method='RK45'
    )
    
    # Extract the wave function u from the solution
    u = sol.y[0]
    
    # Normalize the wave function using Simpson's rule
    norm_factor = integrate.simps(u**2, R)
    ur = u / np.sqrt(norm_factor)
    
    return ur


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
    
    # Solve the Schrödinger equation using the initial guess
    ur = SolveSchroedinger(y0, En, l, R)
    
    # Divide the wavefunction by r^l at the first two grid points
    u_div_r_l_1 = ur[0] / (R[0]**l)
    u_div_r_l_2 = ur[1] / (R[1]**l)
    
    # Perform linear extrapolation to estimate the wavefunction at r=0
    # Using the first two grid points
    f_at_0 = u_div_r_l_1 + (u_div_r_l_2 - u_div_r_l_1) * (0 - R[0]) / (R[1] - R[0])
    
    return f_at_0



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
    Ebnd = []  # Initialize list for bound states

    # Iterate over the energy mesh to find bound states
    for En in Esearch:
        # Extrapolate the wave function at r=0 using the Shoot function
        f_at_0 = Shoot(En, R, l, y0)

        # Check if the extrapolated value is close to zero, indicating a bound state
        if np.isclose(f_at_0, 0, atol=1e-6):
            Ebnd.append((l, En))

        # Break if the desired number of bound states is reached
        if len(Ebnd) >= nmax:
            break

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