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

    # Calculate built-in biases using Simpson's Rule for integral approximation
    def simpsons_log_approximation(x, n_segments=1000):
        """Approximate natural log(x) using Simpson's Rule."""
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        
        f = lambda t: 1 / t
        a, b = 1, x
        h = (b - a) / n_segments
        integral = f(a) + f(b)
        
        for i in range(1, n_segments, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n_segments-1, 2):
            integral += 2 * f(a + i * h)
        
        integral *= h / 3
        return integral

    # Calculate the built-in biases using the Simpson's Rule approximation
    phi_p = V_T * simpsons_log_approximation(N_A / n_i)
    phi_n = V_T * simpsons_log_approximation(N_D / n_i)

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