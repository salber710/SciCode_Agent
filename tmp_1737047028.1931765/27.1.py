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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('27.1', 3)
target = targets[0]

assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)
target = targets[1]

assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)
target = targets[2]

assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)
