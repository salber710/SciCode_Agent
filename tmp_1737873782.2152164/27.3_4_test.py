from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def Fermi(N_A, N_D, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_A: float, doping concentration in p-type region # cm^{-3}
    N_D: float, doping concentration in n-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    
    # Thermal voltage at room temperature (300K)
    V_t = 0.0259  # volts

    # Calculate built-in potential for p-type region
    # phi_p = V_t * ln(N_A / n_i)
    phi_p = V_t * np.log(N_A / n_i)

    # Calculate built-in potential for n-type region
    # phi_n = V_t * ln(N_D / n_i)
    phi_n = V_t * np.log(N_D / n_i)

    return phi_p, phi_n



def capacitance(xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the capacitance of a p-i-n diode.
    Input:
    xi (float): Width of the intrinsic region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density of the material # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the p-i-n diode (V).
    Output:
    C (float): Capacitance of the p-i-n diode (F).
    '''
    
    # Convert xi from micrometers to meters for calculation
    xi_m = xi * 1e-6  # m

    # Convert A from micrometers^2 to meters^2 for calculation
    A_m2 = A * 1e-12  # m^2

    # Vacuum permittivity in F/m
    eps_0 = 8.854e-12  # F/m

    # Calculate the built-in potential using the Fermi function
    phi_p, phi_n = Fermi(N_A, N_D, n_i)

    # Calculate the built-in voltage V_bi
    V_bi = phi_p + phi_n

    # Calculate the depletion width W
    W = xi_m + np.sqrt(2 * es * eps_0 * (V_bi - V0) / (np.abs(N_A - N_D) * 1.6e-19))

    # Calculate the capacitance C using the formula for parallel-plate capacitor
    C = (es * eps_0 * A_m2) / W

    return C



def get_3dB_frequency(R, xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the 3dB frequency of a photodetector.
    Input:
    R (float): Load resistance (Ohms).
    xi (float): Intrinsic width of the depletion region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the PN junction (V).
    Output:
    f_3dB (float): 3dB frequency (Hz).
    '''

    # Calculate the capacitance of the p-i-n diode
    C = capacitance(xi, A, N_A, N_D, n_i, es, V0)

    # Calculate the 3dB frequency using the RC time constant formula
    f_3dB = 1 / (2 * np.pi * R * C)

    return f_3dB


try:
    targets = process_hdf5_to_tuple('27.3', 4)
    target = targets[0]
    xi_arr = np.linspace(0, 5, 50)
    f3dB = get_3dB_frequency(50, xi_arr, 700, 1e19, 1e17, 1.8e6, 13, 0)
    xi_test = np.linspace(0, 5, 50)
    f_test = get_3dB_frequency(50, xi_test, 700, 1e50, 1e50, 1.8e6, 13, 0)
    score = (f3dB - f_test)/f3dB
    assert (np.min(score)==score[-1] and np.max(score)==score[0]) == target

    target = targets[1]
    xi_arr = np.linspace(0, 5, 50)
    assert np.allclose(get_3dB_frequency(50, xi_arr, 1400, 1e16, 1e15,1.8e6,13, 0), target)

    target = targets[2]
    xi_arr = np.linspace(0, 5, 50)
    assert np.allclose(get_3dB_frequency(50, xi_arr, 1400, 1e19, 1e17,1.8e6,13, 0), target)

    target = targets[3]
    xi_arr = np.linspace(0, 5, 50)
    assert np.allclose(get_3dB_frequency(50, xi_arr, 5000, 1e19, 1e17,1.8e6,13, 0), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e