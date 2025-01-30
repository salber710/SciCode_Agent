from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: 
# In semiconductor physics, the built-in potential (or built-in bias) is a key concept in understanding the behavior of p-n junctions.
# The built-in potential is the electric potential difference across the depletion region of a p-n junction in thermal equilibrium.
# It arises due to the difference in the Fermi levels of the n-type and p-type regions.
# The Fermi level is the energy level at which the probability of finding an electron is 50%.
# For n-type semiconductors, the Fermi level is closer to the conduction band, while for p-type semiconductors, it is closer to the valence band.
# The built-in potential can be calculated using the formula:
#   phi_p = V_T * ln(N_a / n_i)
#   phi_n = V_T * ln(N_d / n_i)
# where V_T is the thermal voltage, which is approximately 0.0259 V at room temperature (300 K).
# N_a and N_d are the doping concentrations of the p-type and n-type regions, respectively, and n_i is the intrinsic carrier density.


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

    # Calculate the built-in potential for the p-type region
    phi_p = V_T * np.log(N_a / n_i)

    # Calculate the built-in potential for the n-type region
    phi_n = V_T * np.log(N_d / n_i)

    return phi_p, phi_n


try:
    targets = process_hdf5_to_tuple('34.1', 3)
    target = targets[0]
    assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)

    target = targets[1]
    assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)

    target = targets[2]
    assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e