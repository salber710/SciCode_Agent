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





def Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr):
    '''Calculate the output power and normalized population inversion along the length of the fiber.'''

    # Constants
    speed_of_light = 3e8  # Speed of light in m/s
    planck_constant = 6.626e-34  # Planck's constant in J*s

    # Calculate pump and signal photon energies
    E_signal = planck_constant * speed_of_light / lambda_s
    E_pump = planck_constant * speed_of_light / lambda_p

    # Define spatial domain
    z = np.linspace(0, L, 300)

    # Initial guess for the solution
    y_initial = np.zeros((4, z.size))
    y_initial[0, :] = Ppl  # Forward pump
    y_initial[1, :] = Ppr  # Backward pump
    y_initial[2, :] = 1e-4  # Forward signal
    y_initial[3, :] = 1e-4  # Backward signal

    # Rate equations
    def rate_equations(z, y):
        Pf, Pb, Sf, Sb = y
        dPf_dz = -((sigma_ap - sigma_ep) * N * Pf * gamma_p + alpha_p * Pf)
        dPb_dz = ((sigma_ap - sigma_ep) * N * Pb * gamma_p - alpha_p * Pb)
        dSf_dz = ((sigma_es - sigma_as) * N * Sf * gamma_s - alpha_s * Sf)
        dSb_dz = -((sigma_es - sigma_as) * N * Sb * gamma_s + alpha_s * Sb)
        return np.array([dPf_dz, dPb_dz, dSf_dz, dSb_dz])

    # Boundary conditions
    def boundary_conditions(ya, yb):
        return np.array([ya[0] - Ppl, yb[1] - Ppr, ya[2] - R1 * yb[3], yb[2] - R2 * ya[3]])

    # Solve BVP
    solution = solve_bvp(rate_equations, boundary_conditions, z, y_initial)

    # Output power calculation at the end of the fiber
    output_power = solution.y[2, -1] * (1 - R2)

    # Calculate the normalized population inversion
    Pf = solution.y[0, :]
    Pb = solution.y[1, :]
    Sf = solution.y[2, :]
    Sb = solution.y[3, :]
    inversion_density = N * (Sf + Sb) / ((Sf + Sb) + (Pf + Pb) / (E_pump / (sigma_ap * tau)))
    normalized_inversion = inversion_density / N

    return output_power, normalized_inversion


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