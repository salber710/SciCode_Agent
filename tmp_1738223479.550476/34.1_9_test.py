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
    # Thermal potential at room temperature
    V_T = 0.0259

    # Calculate the natural logarithm using the Newton-Raphson method
    def newton_raphson_ln(x, tolerance=1e-10, max_iterations=100):
        if x <= 0:
            raise ValueError("x must be positive for logarithm")
        guess = x - 1.0
        for _ in range(max_iterations):
            next_guess = guess + 2 * ((x - pow(2, guess)) / (x + pow(2, guess)))
            if abs(next_guess - guess) < tolerance:
                return next_guess
            guess = next_guess
        return guess

    # Calculate phi_p and phi_n using the Newton-Raphson method for natural logarithm
    phi_p = V_T * newton_raphson_ln(N_a / n_i)
    phi_n = V_T * newton_raphson_ln(N_d / n_i)

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