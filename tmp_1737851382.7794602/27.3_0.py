import numpy as np

# Background: 
# In semiconductor physics, the built-in potential (or built-in bias) is the potential difference across a p-n junction in thermal equilibrium. 
# It arises due to the difference in the Fermi levels of the n-type and p-type regions. 
# The built-in potential can be calculated using the formula:
# 
# phi_p = V_T * ln(N_A / n_i)
# phi_n = V_T * ln(N_D / n_i)
# 
# where V_T is the thermal voltage, which is approximately 0.0259 V at room temperature (300 K), 
# N_A is the acceptor concentration in the p-type region, N_D is the donor concentration in the n-type region, 
# and n_i is the intrinsic carrier concentration. 
# The natural logarithm (ln) is used in these calculations.


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

    if n_i <= 0:
        raise ValueError("Intrinsic carrier density n_i must be positive.")
    if N_A <= 0 or N_D <= 0:
        raise ValueError("Doping concentrations N_A and N_D must be positive.")

    # Calculate the built-in bias for the p-type region
    phi_p = V_T * np.log(N_A / n_i)

    # Calculate the built-in bias for the n-type region
    phi_n = V_T * np.log(N_D / n_i)

    return phi_p, phi_n


# Background: 
# In a p-i-n diode, the capacitance is determined by the geometry of the diode and the permittivity of the material.
# The capacitance C of a p-i-n diode can be calculated using the formula:
# C = (ε * A) / (d)
# where ε is the permittivity of the material, A is the area of the diode, and d is the effective width of the depletion region.
# The permittivity ε is given by ε = ε_r * ε_0, where ε_r is the relative permittivity and ε_0 is the vacuum permittivity.
# The effective width of the depletion region d in a p-i-n diode is approximately the width of the intrinsic region xi, 
# since the intrinsic region is fully depleted.
# The capacitance is inversely proportional to the width of the depletion region, meaning that as the width increases, the capacitance decreases.


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
    if xi <= 0:
        raise ValueError("Width of the intrinsic region must be positive.")
    
    # Convert xi from micrometers to meters
    xi_m = xi * 1e-6  # μm to m

    # Convert A from square micrometers to square meters
    A_m2 = A * 1e-12  # μm^2 to m^2

    # Vacuum permittivity in F/m
    epsilon_0 = 8.854e-12

    # Calculate the permittivity of the material
    epsilon = es * epsilon_0

    # Calculate the capacitance
    C = (epsilon * A_m2) / xi_m

    return C



# Background: 
# The 3dB frequency, also known as the cutoff frequency, is the frequency at which the output power of a system is reduced to half of its maximum value.
# In the context of a photodetector, the 3dB frequency is determined by the RC time constant of the circuit, where R is the load resistance and C is the capacitance of the diode.
# The 3dB frequency can be calculated using the formula:
# f_3dB = 1 / (2 * π * R * C)
# where R is the load resistance in ohms, and C is the capacitance in farads.
# The capacitance C can be obtained from the previously defined capacitance function, which depends on the intrinsic width of the depletion region, the detector area, and the permittivity of the material.


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
    
    # Calculate the 3dB frequency
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
