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



def bc(ya, yb, Ppl, Ppr, R1, R2):
    # Define boundary conditions using a complex number approach to simulate phase interactions
    bc = np.zeros(4, dtype=complex)
    bc[0] = (ya[0] - Ppl) * np.exp(1j * np.angle(ya[0]))
    bc[1] = (yb[1] - Ppr) * np.exp(1j * np.angle(yb[1]))
    bc[2] = ya[2] - R1 * (ya[3] * np.exp(1j * np.pi / 4))
    bc[3] = yb[3] - R2 * (yb[2] * np.exp(1j * np.pi / 4))
    return bc



# Background: In fiber lasers, the output power and spatial inversion profile are critical for understanding the laser's performance. 
# The output power is influenced by the gain and loss mechanisms within the fiber, which are determined by the absorption and emission 
# cross-sections, the overlap factors, and the reflectivities of the mirrors. The spatial inversion profile describes the distribution 
# of excited ions along the fiber, which affects the amplification of the signal. The saturation power is a key parameter that 
# determines the intensity at which the gain starts to saturate. The boundary value problem (BVP) approach is used to solve the 
# differential equations governing the evolution of pump and signal intensities, subject to boundary conditions defined by the 
# input and output powers and mirror reflectivities. The speed of light and Planck's constant are used to relate the wavelengths 
# to photon energies, which are essential for calculating the saturation powers.



def Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr):
    '''Calculate the output power and normalized population inversion along the length of the fiber.'''

    # Constants
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s

    # Calculate saturation powers
    Ppsat = h * c / lambda_p / (sigma_ap + sigma_ep) / tau / A_c
    Pssat = h * c / lambda_s / (sigma_as + sigma_es) / tau / A_c

    # Define the spatial domain
    z = np.linspace(0, L, 100)

    # Initial guess for the intensities (Pf, Pb, Sf, Sb)
    y_init = np.zeros((4, z.size))
    y_init[0, :] = Ppl  # Initial guess for forward pump
    y_init[1, :] = Ppr  # Initial guess for backward pump
    y_init[2, :] = 0.1  # Initial guess for forward signal
    y_init[3, :] = 0.1  # Initial guess for backward signal

    # Define the rate equations function
    def f(z, y):
        Pf, Pb, Sf, Sb = y
        Wp = np.log((Pf + Pb) / Ppsat + 1)
        Ws = np.log((Sf + Sb) / Pssat + 1)
        inversion = N * (1 - np.exp(-sigma_ap * Wp - sigma_as * Ws))
        dPf_dz = -gamma_p * (sigma_ap * inversion - sigma_ep * (N - inversion)) * Pf * np.log(1 + Pf / Ppsat)
        dPb_dz = gamma_p * (sigma_ap * inversion - sigma_ep * (N - inversion)) * Pb * np.log(1 + Pb / Ppsat)
        dSf_dz = gamma_s * (sigma_es * inversion - sigma_as * (N - inversion)) * Sf * np.log(1 + Sf / Pssat)
        dSb_dz = -gamma_s * (sigma_es * inversion - sigma_as * (N - inversion)) * Sb * np.log(1 + Sb / Pssat)
        return np.array([dPf_dz, dPb_dz, dSf_dz, dSb_dz])

    # Define the boundary conditions function
    def bc(ya, yb):
        bc = np.zeros(4, dtype=complex)
        bc[0] = (ya[0] - Ppl) * np.exp(1j * np.angle(ya[0]))
        bc[1] = (yb[1] - Ppr) * np.exp(1j * np.angle(yb[1]))
        bc[2] = ya[2] - R1 * (ya[3] * np.exp(1j * np.pi / 4))
        bc[3] = yb[3] - R2 * (yb[2] * np.exp(1j * np.pi / 4))
        return bc

    # Solve the boundary value problem
    sol = solve_bvp(f, bc, z, y_init)

    # Calculate the output power
    Pout = sol.y[2, -1] * (1 - R2)

    # Calculate the normalized population inversion along the fiber
    Pf, Pb, Sf, Sb = sol.y
    Wp = np.log((Pf + Pb) / Ppsat + 1)
    Ws = np.log((Sf + Sb) / Pssat + 1)
    nz = N * (1 - np.exp(-sigma_ap * Wp - sigma_as * Ws)) / N

    return Pout, nz


try:
    targets = process_hdf5_to_tuple('43.3', 4)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    # Define all input parameters
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
    # Call the main function
    assert cmp_tuple_or_list(Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    # Define all input parameters
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
    Ppl = 30 # Left pump power in watts
    Ppr = 50  # Right pump power in watts
    # Call the main function
    assert cmp_tuple_or_list(Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    # Define all input parameters
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
    Ppr = 30  # Right pump power in watts
    # Call the main function
    assert cmp_tuple_or_list(Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr), target)

    target = targets[3]
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
    Ppr = 30  # Right pump power in watts
    # Call the main function
    Pout, nz = Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr)
    assert (nz[0] > nz[len(nz)//2]) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e