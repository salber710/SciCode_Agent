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
