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



# Background: In a p-i-n diode, the intrinsic region is sandwiched between the p-type and n-type regions,
# which affects the diode's capacitance. The capacitance of a p-i-n diode is determined by the geometry
# and material properties of the diode, as well as the applied voltage.
# The capacitance C of the diode can be calculated using the relative permittivity (es), the area (A),
# and the width of the intrinsic region (xi). The formula for capacitance is derived from the parallel-plate
# capacitor model, which assumes the diode acts as a capacitor with the intrinsic region as the dielectric.
# The capacitance C is given by:
# C = (es * epsilon_0 * A) / xi
# where epsilon_0 is the vacuum permittivity (8.854 x 10^-12 F/m) and A is the area in meters squared.
# Since xi is given in micrometers, it should be converted to meters for the calculation.


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
    xi_m = xi * 1e-6  # xi in meters
    A_m2 = A * 1e-12  # A in meters squared
    
    # Vacuum permittivity in F/m
    epsilon_0 = 8.854e-12
    
    # Calculate the capacitance using the parallel-plate capacitor formula
    C = (es * epsilon_0 * A_m2) / xi_m
    
    return C

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('27.2', 3)
target = targets[0]

assert np.allclose(capacitance(5, 1000, 1e10, 1e8,1.8e6,13, 0)*10**15, target)
target = targets[1]

assert np.allclose(capacitance(0.1, 700, 1e19, 1e17,1.8e6,13, 0)*10**15, target)
target = targets[2]

assert np.allclose(capacitance(0.5, 1000, 1e19, 1e17,1.8e6,13, -3)*10**15, target)
