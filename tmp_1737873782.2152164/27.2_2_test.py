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
    
    # Convert xi from micrometers to meters
    xi_m = xi * 1e-6
    
    # Convert A from square micrometers to square meters
    A_m2 = A * 1e-12
    
    # Vacuum permittivity in F/m
    epsilon_0 = 8.854e-12
    
    # Calculate the permittivity of the material
    epsilon = es * epsilon_0
    
    # Calling Fermi function to get built-in potential values
    phi_p, phi_n = Fermi(N_A, N_D, n_i)
    
    # Calculate the built-in voltage
    V_bi = phi_p + phi_n
    
    # Calculate the total capacitance using the formula for a parallel plate capacitor
    C = (epsilon * A_m2) / (xi_m / (1 + (V0 / V_bi)))
    
    return C


try:
    targets = process_hdf5_to_tuple('27.2', 3)
    target = targets[0]
    assert np.allclose(capacitance(5, 1000, 1e10, 1e8,1.8e6,13, 0)*10**15, target)

    target = targets[1]
    assert np.allclose(capacitance(0.1, 700, 1e19, 1e17,1.8e6,13, 0)*10**15, target)

    target = targets[2]
    assert np.allclose(capacitance(0.5, 1000, 1e19, 1e17,1.8e6,13, -3)*10**15, target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e