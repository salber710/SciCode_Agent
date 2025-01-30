from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




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
    kT_q = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the built-in potential
    V_bi = kT_q * np.log((N_a * N_d) / (n_i ** 2))

    # Calculate Fermi levels relative to the intrinsic energy level E_i
    phi_p = kT_q * np.log(N_a / n_i)
    phi_n = kT_q * np.log(N_d / n_i)

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