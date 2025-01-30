from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



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

    # Thermal potential at room temperature in volts
    V_T = 0.0259

    # Calculate the built-in biases using a different method to compute logarithms

    phi_p = V_T * (math.log10(N_A) - math.log10(n_i)) * math.log(math.e, 10)
    phi_n = V_T * (math.log10(N_D) - math.log10(n_i)) * math.log(math.e, 10)

    return phi_p, phi_n


try:
    targets = process_hdf5_to_tuple('27.1', 3)
    target = targets[0]
    assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)

    target = targets[1]
    assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)

    target = targets[2]
    assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e