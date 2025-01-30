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
    
    # Calculate built-in biases using a piecewise linear approximation for logarithms
    def piecewise_log_approx(x):
        """Approximate the natural logarithm using a piecewise linear function."""
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        
        # Define breakpoints and slopes for piecewise linear approximation
        breakpoints = [0.1, 1, 10, 100]
        slopes = [-2, 0, 2, 4]

        if x < breakpoints[0]:
            return slopes[0] * (x - breakpoints[0])
        elif x < breakpoints[1]:
            return slopes[1] * (x - breakpoints[1])
        elif x < breakpoints[2]:
            return slopes[2] * (x - breakpoints[2])
        else:
            return slopes[3] * (x - breakpoints[3])

    # Calculate the built-in biases using the piecewise approximation
    phi_p = V_T * piecewise_log_approx(N_A / n_i)
    phi_n = V_T * piecewise_log_approx(N_D / n_i)

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