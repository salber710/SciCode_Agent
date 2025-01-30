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

    # Use a different approach by employing a custom iterative method to compute logarithms
    def iterative_log(x, base_change_factor=1e-5):
        """Approximate natural log(x) by iteratively adjusting x towards 1."""

        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        
        iter_log = 0
        while x > 1 + base_change_factor:
            x /= math.e
            iter_log += 1
        while x < 1 - base_change_factor:
            x *= math.e
            iter_log -= 1
        
        iter_log += (x - 1) - (x - 1)**2 / 2 + (x - 1)**3 / 3
        return iter_log

    # Calculate the built-in biases using the iterative logarithm approach
    phi_p = V_T * iterative_log(N_A / n_i)
    phi_n = V_T * iterative_log(N_D / n_i)

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