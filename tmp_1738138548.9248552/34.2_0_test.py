from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def Fermi(N_a, N_d, n_i):
    V_T = 0.0259  # Thermal voltage at room temperature in volts

    # Using the hyperbolic sine function for a unique approach
    phi_p = V_T * log((1 + (N_a / n_i)) / (1 - (N_a / n_i))) / 2
    phi_n = V_T * log((1 + (N_d / n_i)) / (1 - (N_d / n_i))) / 2

    return phi_p, phi_n



# Background: 
# The depletion region in a pn-junction is the area around the junction where mobile charge carriers are depleted. 
# The width of the depletion region can be calculated using Poisson's equation, which relates the electric field 
# to the charge density. The depletion width on the n-side (x_n) and the p-side (x_p) can be determined by the 
# following formulas derived from the charge neutrality condition and Poisson's equation:
# 
# x_n = sqrt((2 * ε * V_bi) / (q * (N_a + N_d)) * (N_a / N_d))
# x_p = sqrt((2 * ε * V_bi) / (q * (N_a + N_d)) * (N_d / N_a))
# 
# where:
# - ε is the permittivity of the semiconductor material, calculated as ε = ε_r * ε_0, with ε_0 being the vacuum permittivity.
# - V_bi is the built-in potential, which is the sum of the potentials from the n-type and p-type regions (φ_n + φ_p).
# - q is the electron charge.
# - N_a and N_d are the doping concentrations of the p-type and n-type regions, respectively.
# 
# The vacuum permittivity ε_0 is 8.854 x 10^-14 F/cm, and the electron charge q is 1.6 x 10^-19 C.


def depletion(N_a, N_d, n_i, e_r):
    '''This function calculates the depletion width in both n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    '''
    
    # Constants
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19  # Electron charge in C
    
    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0
    
    # Calculate the built-in potential using the Fermi function
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    V_bi = phi_p + phi_n  # Total built-in potential
    
    # Calculate the depletion widths
    xn = np.sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_a / N_d))
    xp = np.sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_d / N_a))
    
    return xn, xp


try:
    targets = process_hdf5_to_tuple('34.2', 3)
    target = targets[0]
    assert np.allclose(depletion(2*10**17,3*10**17,10**12,15), target)

    target = targets[1]
    assert np.allclose(depletion(1*10**17,2*10**17,10**12,15), target)

    target = targets[2]
    assert np.allclose(depletion(2*10**17,3*10**17,2*10**11,15), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e