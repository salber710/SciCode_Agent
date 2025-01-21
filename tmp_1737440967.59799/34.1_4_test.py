import numpy as np



# Background: 
# In semiconductor physics, the Fermi level is a crucial concept that indicates the energy level at which the probability of finding an electron is 50%. 
# For doped semiconductors, the Fermi levels are shifted from the intrinsic level due to the presence of dopants.
# The built-in potential (also known as the built-in voltage or bias) for n-type and p-type regions can be calculated by comparing the Fermi level with the intrinsic energy level (E_i).
# The thermal voltage (V_T) is kT/q, where k is the Boltzmann constant, T is the temperature, and q is the charge of an electron. At room temperature (approximately 300K), this is approximately 0.0259 V.
# The built-in bias for the p-type region and n-type region can be calculated using the formulas:
# phi_p = V_T * ln(N_a / n_i) for the p-type region
# phi_n = V_T * ln(N_d / n_i) for the n-type region
# Here, ln denotes the natural logarithm.


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
    V_T = 0.0259  # Thermal potential at room temperature in volts

    # Calculate the built-in potentials
    phi_p = V_T * np.log(N_a / n_i)
    phi_n = V_T * np.log(N_d / n_i)

    return phi_p, phi_n

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('34.1', 3)
target = targets[0]

assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)
target = targets[1]

assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)
target = targets[2]

assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)
