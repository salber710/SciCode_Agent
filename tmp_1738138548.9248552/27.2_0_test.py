from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def Fermi(N_A, N_D, n_i):
    V_T = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the built-in potential using the arithmetic mean and logarithm
    phi_p = V_T * ((N_A + n_i) / 2 / n_i) * np.log(N_A / n_i)
    phi_n = V_T * ((N_D + n_i) / 2 / n_i) * np.log(N_D / n_i)

    return phi_p, phi_n



# Background: 
# The capacitance of a p-i-n diode is determined by the geometry of the diode and the permittivity of the material.
# In a p-i-n diode, the intrinsic layer (i-layer) is sandwiched between the p-type and n-type regions. 
# The total capacitance (C) of the diode can be calculated using the formula for a parallel plate capacitor:
# C = ε * A / d, where ε is the permittivity of the material, A is the area of the plates, and d is the separation between the plates.
# For a p-i-n diode, the effective separation is the width of the intrinsic region (x_i).
# The permittivity ε is given by ε = ε_r * ε_0, where ε_r is the relative permittivity and ε_0 is the vacuum permittivity.
# The built-in potential and applied voltage also affect the depletion width and thus the capacitance, but for simplicity, 
# we assume the intrinsic layer dominates the capacitance in a p-i-n diode.

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
    
    # Calculate the capacitance using the formula for a parallel plate capacitor
    C = epsilon * A_m2 / xi_m
    
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