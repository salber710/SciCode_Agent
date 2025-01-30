from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def Fermi(N_A, N_D, n_i):
    V_T = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the built-in potential using the arithmetic mean and logarithm
    phi_p = V_T * ((N_A + n_i) / 2 / n_i) * np.log(N_A / n_i)
    phi_n = V_T * ((N_D + n_i) / 2 / n_i) * np.log(N_D / n_i)

    return phi_p, phi_n


def capacitance(xi, A, N_A, N_D, n_i, es, V0):


    # Constants
    epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m

    # Convert xi from micrometers to meters and A from square micrometers to square meters
    xi_m = xi * 1e-6
    A_m2 = A * 1e-12
    
    # Calculate the permittivity of the material
    epsilon = es * epsilon_0
    
    # Calculate the built-in potential (V_bi) using the intrinsic carrier density and doping concentrations
    kT = 0.0259  # Thermal voltage at room temperature (V)
    V_bi = kT * np.log((N_A * N_D) / n_i**2)
    
    # Calculate the effective width of the depletion region considering the applied voltage
    xi_eff = xi_m * np.sqrt(1 + np.exp(-np.abs(V0) / V_bi))
    
    # Calculate the capacitance using the effective width of the depletion region
    C = epsilon * A_m2 / xi_eff
    
    return C



# Background: The 3dB frequency, also known as the cutoff frequency, is a key parameter in determining the bandwidth of a photodetector. It is the frequency at which the output power drops to half of its maximum value, corresponding to a -3dB point in the frequency response. The 3dB frequency is inversely proportional to the RC time constant of the circuit, where R is the load resistance and C is the capacitance of the device. The formula to calculate the 3dB frequency is f_3dB = 1 / (2 * π * R * C). In this context, the capacitance C is determined by the intrinsic properties of the p-i-n diode, which can be calculated using the previously defined capacitance function.


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
    
    # Calculate the capacitance using the capacitance function
    C = capacitance(xi, A, N_A, N_D, n_i, es, V0)
    
    # Calculate the 3dB frequency using the formula f_3dB = 1 / (2 * π * R * C)
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