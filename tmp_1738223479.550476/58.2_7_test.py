from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density 
    using a continued fraction approximation for exponentiation.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''

    # Continued fraction approximation for power function
    def continued_fraction_power(base, exp, iterations=10):
        # Initialize continued fraction components
        a = [exp] * iterations
        b = [1] * iterations
        b[0] = base  # Start with base as the first b term

        # Compute continued fraction
        result = a[-1]
        for i in range(iterations - 2, -1, -1):
            result = a[i] + b[i] / result

        return base ** exp / result

    # Calculate rho to the power of eos_Gamma using continued fraction approximation
    rho_exp = continued_fraction_power(rho, eos_Gamma)

    # Calculate pressure
    press = eos_kappa * rho_exp

    return press



def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: the density corresponding to the given pressure, a float
    '''

    # Implement a polynomial approximation method to find the density.
    # Here, we'll use a Taylor series expansion approach around an initial estimate.

    # Initial guess using the direct formula
    rho_initial = (press / eos_kappa) ** (1 / eos_Gamma)
    
    # Use a Taylor series expansion to refine the estimate
    # We limit the series to a few terms for demonstration
    delta = 1e-6
    rho_approx = rho_initial
    press_approx = eos_kappa * rho_approx ** eos_Gamma
    
    # Compute the first derivative of the pressure with respect to density
    dP_drho = eos_kappa * eos_Gamma * rho_approx ** (eos_Gamma - 1)
    
    # Compute higher-order terms and adjust the estimate
    for _ in range(3):  # Using a few terms in the expansion
        if dP_drho == 0:
            break
        correction = (press - press_approx) / dP_drho
        rho_approx += correction
        press_approx = eos_kappa * rho_approx ** eos_Gamma
        dP_drho = eos_kappa * eos_Gamma * rho_approx ** (eos_Gamma - 1)

    return rho_approx


try:
    targets = process_hdf5_to_tuple('58.2', 3)
    target = targets[0]
    press = 10
    eos_Gamma = 20
    eos_kappa = 30
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

    target = targets[1]
    press = 1000
    eos_Gamma = 50
    eos_kappa = 80
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

    target = targets[2]
    press = 20000
    eos_Gamma = 2.
    eos_kappa = 100.
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e