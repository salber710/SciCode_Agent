from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import integrate, optimize



def Schroed_deriv(y, r, l, En):
    '''Calculate the derivative of y given r, l and En
    Input 
    y=[u,u'], an list of float where u is the wave function at r, u' is the first derivative of u at r
    r: radius, float
    l: angular momentum quantum number, int
    En: energy, float
    Output
    Schroed: dy/dr=[u',u''] , an 1D array of float where u is the wave function at r, u' is the first derivative of u at r, u'' is the second derivative of u at r
    '''

    # Constants
    Z = 1  # Nuclear charge for hydrogen atom
    a0 = 1  # Bohr radius, set to 1 for simplification

    # Unpack y
    u, up = y

    # Calculate the second derivative of u using the radial part of the Schrödinger equation
    # The equation is u'' = (l*(l+1)/r^2 - 2*Z/r + 2*En) * u
    if r != 0:
        upp = (l * (l + 1) / r**2 - 2 * Z / r + 2 * En) * u
    else:
        # Handle the singularity at r = 0
        upp = 0

    # Return the derivative
    return np.array([up, upp])


try:
    targets = process_hdf5_to_tuple('52.1', 3)
    target = targets[0]
    y = [0.0,-1e-5]
    r = 100
    l = 1
    En = 1
    assert np.allclose(Schroed_deriv(y,r,l,En), target)

    target = targets[1]
    y = [0.0,-2e-5]
    r = 1.1
    l = 2
    En = 1.5
    assert np.allclose(Schroed_deriv(y,r,l,En), target)

    target = targets[2]
    y = [0.0,-2e-5]
    r = 3
    l = 1
    En = 5
    assert np.allclose(Schroed_deriv(y,r,l,En), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e