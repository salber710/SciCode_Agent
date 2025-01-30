from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The Fermi level in a semiconductor is an important concept that helps determine the position of the energy levels in a semiconductor. 
# The built-in bias or potential difference across a pn-junction in a semiconductor is determined by the difference in Fermi levels between 
# the p-type and n-type regions. The built-in bias (φ_p and φ_n) can be calculated using the thermal voltage (V_T), which at room temperature 
# is approximately 0.0259 V. The Fermi level difference for p-type is given by φ_p = V_T * ln(N_a / n_i), and for n-type, it is given by 
# φ_n = V_T * ln(N_d / n_i). These calculations assume the doping concentrations (N_a for acceptors in p-type and N_d for donors in n-type) 
# as well as the intrinsic carrier concentration (n_i) are known.


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

    # Calculate the built-in bias for the p-type and n-type regions
    phi_p = V_T * np.log(N_a / n_i)
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