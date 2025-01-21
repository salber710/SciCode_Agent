import numpy as np

# Background: In semiconductor physics, the built-in potential (also known as built-in bias) 
# is a fundamental concept that describes the potential difference across a p-n junction in 
# thermal equilibrium. This potential arises due to the difference in the concentration of 
# dopants on either side of the junction. The Fermi level, which indicates the chemical 
# potential for electrons, aligns across the junction, resulting in this built-in potential.
# The built-in bias for p-type and n-type regions can be calculated using the thermal voltage 
# (kT/q, often represented as V_T) and the doping concentrations. At room temperature, the 
# thermal voltage is approximately 0.0259 V. The built-in bias for the p-type region (phi_p) 
# and the n-type region (phi_n) can be calculated as follows:
# phi_p = V_T * ln(N_A / n_i)
# phi_n = V_T * ln(N_D / n_i)
# where ln is the natural logarithm, N_A is the doping concentration in the p-type region, 
# N_D is the doping concentration in the n-type region, and n_i is the intrinsic carrier 
# density.


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
    # Thermal potential at room temperature in volts
    V_T = 0.0259
    
    # Calculate the built-in potential for the p-type region
    phi_p = V_T * np.log(N_A / n_i)
    
    # Calculate the built-in potential for the n-type region
    phi_n = V_T * np.log(N_D / n_i)
    
    return phi_p, phi_n


# Background: In semiconductor physics, the capacitance of a p-i-n diode is determined by the 
# depletion region's width and the permittivity of the material. The p-i-n diode has three 
# regions: p-type, intrinsic (undoped), and n-type. The intrinsic region acts as the main 
# space where the depletion region extends under zero or reverse bias, leading to high 
# resistance and low capacitance. The total capacitance is a function of the permittivity 
# of the material, the area of the diode, and the width of the intrinsic region. The 
# capacitance C can be calculated using the formula:
# C = (ε * A) / d
# where ε is the permittivity of the material (ε = ε_r * ε_0), A is the area, and d is the 
# width of the intrinsic region. The permittivity ε is the product of the vacuum permittivity 
# ε_0 and the relative permittivity ε_r. The intrinsic region thickness xi must be converted 
# from micrometers to meters for consistent SI unit calculations.


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
    
    # Convert A from micrometers squared to meters squared
    A_m2 = A * 1e-12
    
    # Vacuum permittivity in F/m
    epsilon_0 = 8.854e-12
    
    # Calculate the permittivity of the material
    epsilon = es * epsilon_0
    
    # Calculate the capacitance using the formula: C = (ε * A) / d
    C = (epsilon * A_m2) / xi_m
    
    return C



# Background: In electronics, the 3dB frequency, also known as the cutoff frequency, is a 
# parameter that indicates the frequency at which the output power drops to half its maximum 
# value, or equivalently, where the output voltage drops to 1/√2 of its maximum value. For a 
# photodetector, this frequency can be determined by the RC time constant of the circuit, 
# where R is the load resistance and C is the capacitance of the photodetector. The 3dB 
# frequency (f_3dB) is given by the formula:
# f_3dB = 1 / (2 * π * R * C)
# where R is the load resistance in ohms and C is the capacitance in farads. The capacitance 
# can be computed using the capacitance function developed earlier, which depends on the 
# permittivity, area, and width of the intrinsic region of the diode.

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
    # Use the capacitance function to calculate the capacitance C
    C = capacitance(xi, A, N_A, N_D, n_i, es, V0)
    
    # Calculate the 3dB frequency using the formula: f_3dB = 1 / (2 * π * R * C)
    f_3dB = 1 / (2 * np.pi * R * C)
    
    return f_3dB

from scicode.parse.parse import process_hdf5_to_tuple
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
