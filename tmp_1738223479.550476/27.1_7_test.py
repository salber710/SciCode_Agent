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

    # Implement logarithm calculation using series expansion for ln(x)
    def series_log(x, terms=10):
        """Approximate natural log(x) using a series expansion."""
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        if x == 1:
            return 0
        elif x > 1:
            x -= 1
            result = 0
            for n in range(1, terms + 1):
                term = ((-1) ** (n + 1)) * (x ** n) / n
                result += term
            return result
        else:
            # Use the identity ln(x) = -ln(1/x) for x < 1
            return -series_log(1/x, terms)

    # Calculate built-in biases using the series expansion log method
    phi_p = V_T * series_log(N_A / n_i)
    phi_n = V_T * series_log(N_D / n_i)

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