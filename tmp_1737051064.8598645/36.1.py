import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton



# Background: 
# The generation of electron-hole pairs in a semiconductor due to incident light is a fundamental concept in optoelectronics. 
# When light with a certain wavelength (λ) is incident on a semiconductor, it can be absorbed, generating electron-hole pairs. 
# The rate of generation of these carriers depends on the optical power (P), the area over which the light is incident (A), 
# the absorption coefficient (α), and the depth (x) into the material. The absorption coefficient α indicates how quickly 
# the light intensity decreases as it penetrates the material. The generation rate of carriers is also influenced by the 
# lifetime (τ) of the carriers, which is the average time an electron or hole exists before recombination. 
# The energy of the incident photons is given by E = h*c/λ, where h is Planck's constant and c is the speed of light. 
# The number of photons incident per second is P/E, and the number of photons absorbed per unit volume per second at a 
# depth x is given by (P/E) * (α/A) * exp(-α*x). The generated carrier density n(x) is then this rate multiplied by the 
# carrier lifetime τ.

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

    # Convert units
    lambda_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    A_m2 = A * 1e-12  # Convert area from μm^2 to m^2
    alpha_m = alpha * 1e2  # Convert absorption coefficient from cm^-1 to m^-1
    x_m = x * 1e-6  # Convert depth from μm to m

    # Calculate photon energy
    E_photon = h * c / lambda_m  # Energy of a single photon in Joules

    # Calculate photon flux (number of photons per second per unit area)
    photon_flux = P / E_photon  # Total number of photons per second

    # Calculate generation rate at depth x
    G_x = (photon_flux / A_m2) * alpha_m * np.exp(-alpha_m * x_m)  # Generation rate in m^-3·s^-1

    # Calculate generated carrier density
    dN = G_x * tau  # Carrier density in m^-3

    # Convert carrier density to cm^-3
    dN_cm3 = dN * 1e-6

    return dN_cm3


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('36.1', 3)
target = targets[0]

assert np.allclose(generation(1e-3, 50, 519, 1e4, 1e-9, 1), target)
target = targets[1]

assert np.allclose(generation(10e-3, 50, 519, 1e4, 1e-9, 1), target)
target = targets[2]

assert np.allclose(generation(100e-3, 50, 519, 1e4, 1e-9, 1), target)
