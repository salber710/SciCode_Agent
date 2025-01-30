from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton



# Background: 
# The generation of electron distribution as a function of depth in a semiconductor material is influenced by the absorption of incident light. 
# The number of photons absorbed per unit volume per unit time is given by the product of the absorption coefficient (alpha) and the photon flux.
# The photon flux can be calculated from the incident optical power (P) divided by the energy of a single photon and the beam area (A).
# The energy of a photon is given by E = h * c / lambda, where h is the Planck constant, c is the speed of light, and lambda is the wavelength.
# The generation rate of electron-hole pairs is then modulated by the exponential decay of light intensity with depth, described by the Beer-Lambert law: I(x) = I0 * exp(-alpha * x).
# The generated carrier density (dN) is the product of the generation rate and the carrier lifetime (tau).

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
    c = 3e8  # speed of light in m/s
    h = 6.626e-34  # Planck constant in J·s

    # Convert units
    lambda_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    A_m2 = A * 1e-12  # Convert area from μm^2 to m^2
    alpha_m = alpha * 1e2  # Convert absorption coefficient from cm^-1 to m^-1
    x_m = x * 1e-6  # Convert depth from μm to m

    # Calculate photon energy
    E_photon = h * c / lambda_m  # Energy of a single photon in Joules

    # Calculate photon flux (photons per second per m^2)
    photon_flux = P / (E_photon * A_m2)

    # Calculate generation rate at depth x
    G_x = photon_flux * alpha_m * np.exp(-alpha_m * x_m)  # Generation rate in m^-3·s^-1

    # Calculate generated carrier density
    dN = G_x * tau  # Carrier density in m^-3

    # Convert carrier density to cm^-3
    dN_cm3 = dN * 1e-6

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