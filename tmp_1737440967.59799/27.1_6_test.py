import numpy as np



# Background: 
# In semiconductor physics, the built-in potential (or bias) is a crucial concept for understanding the behavior of pn-junctions. 
# The built-in potential is the potential difference between the p-type and n-type regions at equilibrium. 
# This potential arises due to the difference in concentrations of the electrons and holes in the two regions. 
# The Fermi level in a semiconductor is the energy level at which the probability of finding an electron is 50%. 
# For n-type semiconductors, the Fermi level is closer to the conduction band, while for p-type, it is closer to the valence band.
# The thermal potential (V_T) at room temperature (approximately 300K) is given as 0.0259V.
# The Fermi potential for n-type (phi_n) and p-type (phi_p) can be calculated using the doping concentrations (N_D for n-type, N_A for p-type),
# the intrinsic carrier density (n_i), and the thermal potential (V_T) using the following formulas:
# phi_n = V_T * ln(N_D / n_i)
# phi_p = V_T * ln(N_A / n_i)
# These equations come from the relationship between the Fermi level and the intrinsic energy level (E_i) in doped semiconductors.


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
    phi_n = V_T * np.log(N_D / n_i)
    phi_p = V_T * np.log(N_A / n_i)
    
    return phi_p, phi_n

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('27.1', 3)
target = targets[0]

assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)
target = targets[1]

assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)
target = targets[2]

assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)
