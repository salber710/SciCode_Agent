from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton



def generation(P, A, lambda_i, alpha, tau, x):
    '''This function computes the excess electron distribution.
    Input:
    P (float): incident optical power in W
    A (float): beam area in μm^2
    lambda_i (float): incident wavelength in nm
    alpha (float): absorption coefficient in cm^-1
    tau (float): lifetime of excess carriers in s
    x (float): depth variable in μm
    Output:
    dN (float): generated carrier density in cm^-3
    '''

    # Constants
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J·s
    q = 1.602e-19  # Elementary charge in C

    # Convert units
    lambda_i_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    A_m2 = A * 1e-12  # Convert area from μm^2 to m^2
    alpha_m1 = alpha * 100  # Convert absorption coefficient from cm^-1 to m^-1
    x_m = x * 1e-6  # Convert depth from μm to m

    # Calculate photon energy
    E_photon = h * c / lambda_i_m  # in Joules

    # Calculate photon flux
    photon_flux = P / (E_photon * A_m2)  # in photons/s·m^2

    # Calculate excess carrier generation rate
    G = photon_flux * alpha_m1 * np.exp(-alpha_m1 * x_m)  # in carriers/m^3/s

    # Calculate excess carrier density
    dN = G * tau  # in carriers/m^3

    # Convert excess carrier density to cm^-3
    dN_cm3 = dN * 1e-6  # Convert from m^-3 to cm^-3

    return dN_cm3


try:
    targets = process_hdf5_to_tuple('36.1', 3)
    target = targets[0]
    assert np.allclose(generation(1e-3, 50, 519, 1e4, 1e-9, 1), target)

    target = targets[1]
    assert np.allclose(generation(10e-3, 50, 519, 1e4, 1e-9, 1), target)

    target = targets[2]
    assert np.allclose(generation(100e-3, 50, 519, 1e4, 1e-9, 1), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e