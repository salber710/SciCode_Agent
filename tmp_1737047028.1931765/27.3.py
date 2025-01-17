import numpy as np

# Background: In semiconductor physics, the built-in potential (or built-in bias) is a key concept in understanding the behavior of p-n junctions. 
# It arises due to the difference in Fermi levels between the n-type and p-type regions. The Fermi level is the energy level at which the probability 
# of finding an electron is 50%. In thermal equilibrium, the Fermi level is constant throughout the semiconductor. 
# The built-in potential can be calculated using the doping concentrations of the n-type (N_D) and p-type (N_A) regions, 
# as well as the intrinsic carrier concentration (n_i). The thermal voltage (V_T) at room temperature is approximately 0.0259 V. 
# The built-in potential for the p-type region (phi_p) and the n-type region (phi_n) can be calculated using the following formulas:
# phi_p = V_T * ln(N_A / n_i)
# phi_n = V_T * ln(N_D / n_i)
# These equations are derived from the relationship between the Fermi level and the carrier concentrations in the semiconductor.


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
    V_T = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the built-in potential for the p-type region
    phi_p = V_T * np.log(N_A / n_i)

    # Calculate the built-in potential for the n-type region
    phi_n = V_T * np.log(N_D / n_i)

    return phi_p, phi_n


# Background: In a p-i-n diode, the intrinsic layer is sandwiched between the p-type and n-type regions. 
# The capacitance of a p-i-n diode is influenced by the width of the intrinsic region, the area of the diode, 
# and the permittivity of the material. The capacitance can be calculated using the formula for a parallel plate capacitor, 
# which is C = ε * A / d, where ε is the permittivity, A is the area, and d is the separation between the plates (or the width of the intrinsic region in this case).
# The permittivity ε is given by ε = ε_r * ε_0, where ε_r is the relative permittivity and ε_0 is the vacuum permittivity.
# The built-in potential and the applied voltage also affect the depletion width and thus the capacitance.
# The total capacitance of the p-i-n diode can be calculated considering the intrinsic layer thickness and the applied voltage.


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
    # Convert xi and A from micrometers to meters
    xi_m = xi * 1e-6
    A_m2 = A * 1e-12

    # Vacuum permittivity in F/m
    epsilon_0 = 8.854e-12

    # Calculate the permittivity of the material
    epsilon = es * epsilon_0

    # Calculate the capacitance using the formula for a parallel plate capacitor
    C = epsilon * A_m2 / xi_m

    return C



# Background: The 3dB frequency, also known as the cutoff frequency, is a key parameter in determining the bandwidth of a photodetector. 
# It is the frequency at which the output power drops to half of its maximum value, corresponding to a -3dB point in the frequency response. 
# The 3dB frequency is influenced by the RC time constant of the circuit, where R is the load resistance and C is the capacitance of the device. 
# The RC time constant (τ) is given by τ = R * C. The 3dB frequency (f_3dB) can be calculated using the formula f_3dB = 1 / (2 * π * τ). 
# In this context, the capacitance C is determined by the intrinsic properties of the p-i-n diode, which can be calculated using the capacitance function.


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
    
    # Calculate the RC time constant
    tau = R * C
    
    # Calculate the 3dB frequency
    f_3dB = 1 / (2 * np.pi * tau)
    
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
