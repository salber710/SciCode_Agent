import numpy as np



# Background: 
# In semiconductor physics, the built-in potential (or built-in bias) is the potential difference across a p-n junction in thermal equilibrium. 
# It arises due to the difference in the Fermi levels of the n-type and p-type regions. 
# The built-in potential can be calculated using the doping concentrations of the n-type (N_d) and p-type (N_a) regions, 
# as well as the intrinsic carrier density (n_i). 
# The formula for the built-in potential (V_bi) is given by:
# V_bi = V_t * ln((N_a * N_d) / n_i^2)
# where V_t is the thermal voltage, which is approximately 0.0259 V at room temperature (300 K).
# The built-in bias for the p-type region (phi_p) and the n-type region (phi_n) can be calculated as:
# phi_p = V_bi / 2
# phi_n = -V_bi / 2
# These represent the energy level differences compared to the intrinsic energy level (E_i).


def Fermi(N_a, N_d, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    V_t = 0.0259  # Thermal voltage at room temperature in volts
    V_bi = V_t * np.log((N_a * N_d) / n_i**2)
    phi_p = V_bi / 2
    phi_n = -V_bi / 2

    return phi_p, phi_n


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('34.1', 3)
target = targets[0]

assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)
target = targets[1]

assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)
target = targets[2]

assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)
