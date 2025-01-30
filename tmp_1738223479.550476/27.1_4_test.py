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

    # Thermal potential at room temperature
    V_T = 0.0259

    # Compute the built-in bias using a Taylor series approximation for small arguments
    def taylor_log_approx(x):
        """Approximate natural log(1+x) using a Taylor series expansion."""
        if x == 0:
            return 0
        elif -0.5 < x < 0.5:
            # Use the series: log(1+x) ≈ x - x^2/2 + x^3/3 for |x| < 1
            return x - x*x/2 + x*x*x/3
        else:
            # Fall back to natural logarithm for larger values

            return math.log(1 + x)

    # Calculate built-in biases using the Taylor series approximation
    phi_p = V_T * taylor_log_approx((N_A - n_i) / n_i)
    phi_n = V_T * taylor_log_approx((N_D - n_i) / n_i)

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