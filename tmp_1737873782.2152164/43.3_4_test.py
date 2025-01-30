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
    The rate of change of the power values for the pump and signal along the fiber:
        dydz[0]: Rate of change of forward pump power.
        dydz[1]: Rate of change of backward pump power.
        dydz[2]: Rate of change of forward signal power.
        dydz[3]: Rate of change of backward signal power.
    '''
    
    # Extract power values from y
    Pfwd_pump = y[0]
    Pbwd_pump = y[1]
    Pfwd_signal = y[2]
    Pbwd_signal = y[3]
    
    # Calculate the inversion density of the ions
    Wp_fwd = Pfwd_pump / Ppsat
    Wp_bwd = Pbwd_pump / Ppsat
    Ws_fwd = Pfwd_signal / Pssat
    Ws_bwd = Pbwd_signal / Pssat
    
    # Calculate the rate of change of forward and backward pump powers
    dPfwd_pump_dz = - (sigma_ap * N * (Wp_fwd / (1 + Wp_fwd + Ws_fwd)) * gamma_p + alpha_p) * Pfwd_pump
    dPbwd_pump_dz = (sigma_ap * N * (Wp_bwd / (1 + Wp_bwd + Ws_bwd)) * gamma_p - alpha_p) * Pbwd_pump
    
    # Calculate the rate of change of forward and backward signal powers
    dPfwd_signal_dz = (sigma_es * N * (Ws_fwd / (1 + Wp_fwd + Ws_fwd)) * gamma_s - alpha_s) * Pfwd_signal
    dPbwd_signal_dz = - (sigma_es * N * (Ws_bwd / (1 + Wp_bwd + Ws_bwd)) * gamma_s + alpha_s) * Pbwd_signal
    
    # Pack the rates of change into an array
    dydz = np.array([dPfwd_pump_dz, dPbwd_pump_dz, dPfwd_signal_dz, dPbwd_signal_dz])
    
    return dydz


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
        bc[0]: Difference between the initial forward pump power and left pump power.
        bc[1]: Difference between the final backward pump power and right pump power.
        bc[2]: Difference between the initial forward signal power and the reflected backward signal power.
        bc[3]: Difference between the final backward signal power and the reflected forward signal power.
    '''

    # Boundary condition for the forward pump at the start of the fiber
    bc_0 = ya[0] - Ppl
    
    # Boundary condition for the backward pump at the end of the fiber
    bc_1 = yb[1] - Ppr
    
    # Boundary condition for the forward signal at the start of the fiber
    bc_2 = ya[2] - R1 * ya[3]
    
    # Boundary condition for the backward signal at the end of the fiber
    bc_3 = yb[3] - R2 * yb[2]
    
    return np.array([bc_0, bc_1, bc_2, bc_3])





def Pout_Nz_Calculation(lambda_s, lambda_p, tau, sigma_ap, sigma_ep, sigma_as, sigma_es, A_c, N, alpha_p, alpha_s, gamma_s, gamma_p, R1, R2, L, Ppl, Ppr):
    '''Calculate the output power and normalized population inversion along the length of the fiber.
    Parameters:
    lambda_s : float
        Wavelength of the signal in meters.
    lambda_p : float
        Wavelength of the pump in meters.
    tau : float
        Lifetime of the excited state in seconds.
    sigma_ap : float
        Absorption cross-section for the pump.
    sigma_ep : float
        Emission cross-section for the pump.
    sigma_as : float
        Absorption cross-section for the signal.
    sigma_es : float
        Emission cross-section for the signal.
    A_c : float
        Core area of the fiber in square meters.
    N : float
        Total ion population in the fiber.
    alpha_p : float
        Loss coefficient for the pump.
    alpha_s : float
        Loss coefficient for the signal.
    gamma_s : float
        Gain coefficient for the signal.
    gamma_p : float
        Gain coefficient for the pump.
    R1 : float
        Reflectivity of the input mirror.
    R2 : float
        Reflectivity of the output mirror.
    L : float
        Length of the fiber in meters.
    Ppl : float
        Input power for the left pump.
    Ppr : float
        Input power for the right pump.
    Returns:
    tuple (float, ndarray)
        Pout : float
            Output power of the signal at the end of the fiber, considering the output mirror losses.
        nz : ndarray
            Normalized population inversion distribution along the length of the fiber.
    '''

    # Speed of light and Planck's constant
    c = 3e8  # m/s
    h = 6.626e-34  # J*s

    # Calculate the saturation powers
    Ppsat = h * c / lambda_p / (sigma_ap + sigma_ep) / A_c / tau
    Pssat = h * c / lambda_s / (sigma_as + sigma_es) / A_c / tau
    
    # Define the spatial domain
    z = np.linspace(0, L, 100)  # 100 points along the fiber

    # Initial guess for the intensities [Pfwd_pump, Pbwd_pump, Pfwd_signal, Pbwd_signal]
    y_init = np.zeros((4, z.size))
    y_init[0, :] = Ppl  # Forward pump initial guess
    y_init[1, :] = Ppr  # Backward pump initial guess
    y_init[2, :] = 0.01  # Forward signal initial guess
    y_init[3, :] = 0.01  # Backward signal initial guess
    
    # Solve the boundary value problem
    sol = solve_bvp(lambda z, y: f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s), 
                    lambda ya, yb: bc(ya, yb, Ppl, Ppr, R1, R2), 
                    z, y_init)

    # Extract the solutions for the forward signal power
    Pfwd_signal = sol.sol(z)[2]

    # Calculate normalized population inversion
    Wp_fwd = sol.sol(z)[0] / Ppsat
    Ws_fwd = Pfwd_signal / Pssat
    nz = Wp_fwd / (1 + Wp_fwd + Ws_fwd)

    # Calculate the output power
    Pout = Pfwd_signal[-1] * (1 - R2)

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