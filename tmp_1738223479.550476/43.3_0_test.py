from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_bvp


def f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s):
    '''System of differential equations representing the rate equations of the fiber laser.
    Parameters:
    z : float
        Spatial variable along the fiber's length, representing the position.
    y : ndarray
        Array containing the power values of [forward pump, backward pump, forward signal, backward signal].
    Pssat : float
        Saturation power for the signal.
    Ppsat : float
        Saturation power for the pump.
    N : float
        Total ion population in the fiber.
    sigma_ap : float
        Absorption cross-section for the pump.
    sigma_ep : float
        Emission cross-section for the pump.
    sigma_as : float
        Absorption cross-section for the signal.
    sigma_es : float
        Emission cross-section for the signal.
    gamma_p : float
        Gain coefficient for the pump.
    alpha_p : float
        Loss coefficient for the pump.
    gamma_s : float
        Gain coefficient for the signal.
    alpha_s : float
        Loss coefficient for the signal.
    Returns: ndarray
    The rate of change of the power values for the pump and signal along the fiber.
    '''
    
    # Unpack the power values
    Pf, Pb, Sf, Sb = y
    
    # Define a novel inversion density calculation using a geometric mean approach
    intensity_product = (Pf * Pb * Sf * Sb) ** 0.25
    effective_inv = N * intensity_product / (1 + Pssat * Ppsat / (Pf + Pb + Sf + Sb + 1e-12))
    
    # Calculate the rate of change using the new inversion density model
    dPfdz = -alpha_p * Pf + gamma_p * (sigma_ep * effective_inv - sigma_ap * (N - effective_inv)) * Pf
    dPbdz = alpha_p * Pb - gamma_p * (sigma_ep * effective_inv - sigma_ap * (N - effective_inv)) * Pb
    dSfdz = -alpha_s * Sf + gamma_s * (sigma_es * effective_inv - sigma_as * (N - effective_inv)) * Sf
    dSbdz = alpha_s * Sb - gamma_s * (sigma_es * effective_inv - sigma_as * (N - effective_inv)) * Sb
    
    # Return the rate of changes as a numpy array
    dydz = np.array([dPfdz, dPbdz, dSfdz, dSbdz])
    
    return dydz



def bc(ya, yb, Ppl, Ppr, R1, R2):
    '''Define the boundary conditions for the fiber laser.

    Parameters:
    ya : ndarray
        Array of power values at the start of the fiber.
    yb : ndarray
        Array of power values at the end of the fiber.
    Ppl : float
        Input power for the left pump.
    Ppr : float
        Input power for the right pump.
    R1 : float
        Reflectivity of the input mirror.
    R2 : float
        Reflectivity of the output mirror.

    Returns:
    ndarray
        An array of four boundary conditions.
    '''

    # Distinctive approach: Use polynomial transformations for boundary conditions
    bc = np.array([
        (ya[0] - Ppl)**3,                    # Cubic difference for forward pump power
        (yb[1] - Ppr)**3,                    # Cubic difference for backward pump power
        (ya[2]**3 - R1**3 * ya[3]**3),       # Cubic reflectivity condition at fiber start
        (yb[3]**3 - R2**3 * yb[2]**3)        # Cubic reflectivity condition at fiber end
    ])
    
    return bc



# Background: In fiber lasers, understanding the output power and spatial inversion profile is crucial for optimizing performance. 
# The spatial inversion profile indicates the distribution of excited ions along the fiber, impacting the laser's gain and efficiency.
# The output power is determined by the interaction of the signal and pump wavelengths with the fiber's active medium, influenced by cross-sections, 
# fiber core area, and mirror reflectivities. The saturation power is a key parameter indicating when the gain begins to saturate. 
# Solving the boundary value problem (BVP) with rate equations and boundary conditions allows for calculating these profiles accurately.
# The constants used are the speed of light (3e8 m/s) and Planck's constant (6.626e-34 J*s), which are fundamental in calculating photon energies.



def Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr):
    '''Calculate the output power and normalized population inversion along the length of the fiber.'''

    # Constants
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s

    # Calculate saturation powers
    Pssat = h * c / (lambda_s * sigma_es * tau * A_c)
    Ppsat = h * c / (lambda_p * sigma_ep * tau * A_c)

    # Define the spatial domain
    z = np.linspace(0, L, 100)

    # Initial guess for the power along the fiber
    y_initial = np.zeros((4, z.size))
    y_initial[0, :] = Ppl  # Forward pump
    y_initial[1, :] = Ppr  # Backward pump
    y_initial[2, :] = 1e-3  # Forward signal
    y_initial[3, :] = 1e-3  # Backward signal

    # Solve the boundary value problem
    sol = solve_bvp(lambda z, y: f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s),
                    lambda ya, yb: bc(ya, yb, Ppl, Ppr, R1, R2),
                    z, y_initial)

    # Calculate output power and inversion profile
    Pf_out = sol.y[2, -1]  # Forward signal power at the end
    Pout = Pf_out * (1 - R2)  # Output power considering output mirror losses

    # Calculate population inversion along the fiber
    Pf = sol.y[0, :]
    Pb = sol.y[1, :]
    Sf = sol.y[2, :]
    Sb = sol.y[3, :]
    intensity_product = (Pf * Pb * Sf * Sb) ** 0.25
    effective_inv = N * intensity_product / (1 + Pssat * Ppsat / (Pf + Pb + Sf + Sb + 1e-12))
    nz = effective_inv / N  # Normalized population inversion

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