import numpy as np



# Background: 
# In semiconductor physics, the built-in potential (also known as built-in bias) is a key concept when analyzing p-n junctions.
# It is the electric potential difference between the n-type and p-type regions in equilibrium, and it arises due to the
# difference in doping concentrations between these regions. The built-in potential helps to form the electric field across
# the junction, which is crucial for the operation of semiconductor devices like diodes and transistors.
# 
# The Fermi level in a semiconductor indicates the energy level at which the probability of finding an electron is 50%.
# When a semiconductor is doped, the Fermi level shifts depending on the type and concentration of dopants.
# 
# The built-in potential components, phi_p and phi_n, can be expressed as:
# phi_p = V_T * ln(N_A / n_i) for the p-type region
# phi_n = V_T * ln(N_D / n_i) for the n-type region
# where V_T is the thermal voltage (approximately 0.0259 V at room temperature), N_A is the acceptor concentration,
# N_D is the donor concentration, and n_i is the intrinsic carrier density.


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
    # Thermal voltage at room temperature
    V_T = 0.0259

    # Calculate the built-in bias for the p-type region
    phi_p = V_T * np.log(N_A / n_i)

    # Calculate the built-in bias for the n-type region
    phi_n = V_T * np.log(N_D / n_i)

    return phi_p, phi_n

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('27.1', 3)
target = targets[0]

assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)
target = targets[1]

assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)
target = targets[2]

assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)
