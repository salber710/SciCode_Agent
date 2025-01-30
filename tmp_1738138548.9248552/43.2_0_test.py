from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_bvp


def f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s):
    Pf, Pb, Sf, Sb = y

    # Calculate saturation effects using a logarithmic model
    Wp = np.log((Pf + Pb) / Ppsat + 1)
    Ws = np.log((Sf + Sb) / Pssat + 1)

    # Population inversion using a logarithmic approach
    inversion = N * (1 - np.exp(-sigma_ap * Wp - sigma_as * Ws))

    # Differential equations for pump and signal using logarithmic terms for gain and absorption
    dPf_dz = -gamma_p * (sigma_ap * inversion - sigma_ep * (N - inversion)) * Pf * np.log(1 + Pf / Ppsat)
    dPb_dz = gamma_p * (sigma_ap * inversion - sigma_ep * (N - inversion)) * Pb * np.log(1 + Pb / Ppsat)
    dSf_dz = gamma_s * (sigma_es * inversion - sigma_as * (N - inversion)) * Sf * np.log(1 + Sf / Pssat)
    dSb_dz = -gamma_s * (sigma_es * inversion - sigma_as * (N - inversion)) * Sb * np.log(1 + Sb / Pssat)

    return np.array([dPf_dz, dPb_dz, dSf_dz, dSb_dz])



# Background: In fiber laser systems, boundary conditions are crucial for defining how light behaves at the ends of the fiber. 
# These conditions ensure that the power levels of the pump and signal at the start and end of the fiber are consistent with 
# the physical setup of the laser. The reflectivities of the input and output mirrors (R1 and R2) determine how much light is 
# reflected back into the fiber, affecting the laser's efficiency and stability. The boundary conditions for the forward and 
# backward pump and signal powers are defined by the differences between the actual power values at the fiber's boundaries 
# and the expected values based on the input powers and mirror reflectivities.

def bc(ya, yb, Ppl, Ppr, R1, R2):
    '''Define the boundary conditions for the fiber laser.
    Parameters:
    ya : ndarray
        Array of power values at the start of the fiber. Contains values corresponding to:
        ya[0] - Power of the forward pump at the fiber input.
        ya[1] - Power of the backward pump at the fiber input.
        ya[2] - Power of the forward signal at the fiber input.
        ya[3] - Power of the backward signal at the fiber input.
    yb : ndarray
        Array of power values at the end of the fiber. Contains values corresponding to:
        yb[0] - Power of the forward pump at the fiber output.
        yb[1] - Power of the backward pump at the fiber output.
        yb[2] - Power of the forward signal at the fiber output.
        yb[3] - Power of the backward signal at the fiber output.
    Ppl : float
        Input power for the left pump, affecting the starting boundary of the laser.
    Ppr : float
        Input power for the right pump, affecting the ending boundary of the laser.
    R1 : float
        Reflectivity of the input mirror, modifying the behavior of the light at the fiber's start.
    R2 : float
        Reflectivity of the output mirror, modifying the behavior of the light at the fiber's end.
    Returns:
    ndarray
        An array of four boundary conditions calculated as follows:
        bc[0]: boundary condition for rate of change of forward pump power.
        bc[1]: boundary condition for rate of change of backward pump power.
        bc[2]: boundary condition for rate of change of forward signal power.
        bc[3]: boundary condition for rate of change of backward signal power.
    '''
    # Calculate the boundary conditions
    bc = np.zeros(4)
    bc[0] = ya[0] - Ppl  # Difference between initial forward pump power and left pump power
    bc[1] = yb[1] - Ppr  # Difference between final backward pump power and right pump power
    bc[2] = ya[2] - R1 * ya[3]  # Reflectivity condition at the input mirror for the signal
    bc[3] = yb[3] - R2 * yb[2]  # Reflectivity condition at the output mirror for the signal

    return bc


try:
    targets = process_hdf5_to_tuple('43.2', 3)
    target = targets[0]
    lambda_s = 1100e-9  # Signal wavelength in meters
    lambda_p = 974e-9  # Pump wavelength in meters
    tau = 0.8e-3  # Lifetime in seconds
    sigma_ap = 26e-21 * 1e-4  # Absorption cross-section for pump in square meters
    sigma_ep = 26e-21 * 1e-4  # Emission cross-section for pump in square meters
    sigma_as = 1e-23 * 1e-4  # Absorption cross-section for signal in square meters
    sigma_es = 1.6e-21 * 1e-4  # Emission cross-section for signal in square meters
    A_c = 3.1416e-10  # Core area in square meters
    N = 5.5351e25  # Ion concentration in ions/m^3
    alpha_p = 2e-3  # Pump loss coefficient in 1/m
    alpha_s = 4e-4  # Signal loss coefficient in 1/m
    gamma_s = 0.82  # Fraction of the signal mode overlap with the ions
    gamma_p = 0.0024  # Fraction of the pump mode overlap with the ions
    R1 = 0.99  # Reflectivity of the input mirror
    R2 = 0.035  # Reflectivity of the output mirror
    L = 40  # Fiber length in meters
    Ppl = 50 # Left pump power in watts
    Ppr = 50  # Right pump power in watts
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s
    nu_s = c / lambda_s
    nu_p = c / lambda_p
    ya = [0,0,0,0]
    yb= [0, 0, 0, 0]
    assert np.allclose(bc(ya, yb, Ppl, Ppr, R1, R2), target)

    target = targets[1]
    lambda_s = 1100e-9  # Signal wavelength in meters
    lambda_p = 974e-9  # Pump wavelength in meters
    tau = 0.8e-3  # Lifetime in seconds
    sigma_ap = 26e-21 * 1e-4  # Absorption cross-section for pump in square meters
    sigma_ep = 26e-21 * 1e-4  # Emission cross-section for pump in square meters
    sigma_as = 1e-23 * 1e-4  # Absorption cross-section for signal in square meters
    sigma_es = 1.6e-21 * 1e-4  # Emission cross-section for signal in square meters
    A_c = 3.1416e-10  # Core area in square meters
    N = 5.5351e25  # Ion concentration in ions/m^3
    alpha_p = 2e-3  # Pump loss coefficient in 1/m
    alpha_s = 4e-4  # Signal loss coefficient in 1/m
    gamma_s = 0.82  # Fraction of the signal mode overlap with the ions
    gamma_p = 0.0024  # Fraction of the pump mode overlap with the ions
    R1 = 0.99  # Reflectivity of the input mirror
    R2 = 0.035  # Reflectivity of the output mirror
    L = 40  # Fiber length in meters
    Ppl = 50 # Left pump power in watts
    Ppr = 50  # Right pump power in watts
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s
    nu_s = c / lambda_s
    nu_p = c / lambda_p
    ya = [1, 0.5, 2, 1]
    yb = [0.5, 1, 1, 2]
    assert np.allclose(bc(ya, yb, Ppl, Ppr, R1, R2), target)

    target = targets[2]
    lambda_s = 1100e-9  # Signal wavelength in meters
    lambda_p = 974e-9  # Pump wavelength in meters
    tau = 0.8e-3  # Lifetime in seconds
    sigma_ap = 26e-21 * 1e-4  # Absorption cross-section for pump in square meters
    sigma_ep = 26e-21 * 1e-4  # Emission cross-section for pump in square meters
    sigma_as = 1e-23 * 1e-4  # Absorption cross-section for signal in square meters
    sigma_es = 1.6e-21 * 1e-4  # Emission cross-section for signal in square meters
    A_c = 3.1416e-10  # Core area in square meters
    N = 5.5351e25  # Ion concentration in ions/m^3
    alpha_p = 2e-3  # Pump loss coefficient in 1/m
    alpha_s = 4e-4  # Signal loss coefficient in 1/m
    gamma_s = 0.82  # Fraction of the signal mode overlap with the ions
    gamma_p = 0.0024  # Fraction of the pump mode overlap with the ions
    R1 = 0.99  # Reflectivity of the input mirror
    R2 = 0.035  # Reflectivity of the output mirror
    L = 40  # Fiber length in meters
    Ppl = 50 # Left pump power in watts
    Ppr = 50  # Right pump power in watts
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s
    nu_s = c / lambda_s
    nu_p = c / lambda_p
    ya = [1, 1, 2, 1]
    yb = [0.5, 1, 2, 2]
    assert np.allclose(bc(ya, yb, Ppl, Ppr, R1, R2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e