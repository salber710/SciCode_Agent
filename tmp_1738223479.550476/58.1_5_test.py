from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy as sp
import scipy.integrate as si



def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density using a series expansion approach.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''
    
    # Use a series expansion for the power function for small fractional exponents
    def series_power(base, exp, terms=10):
        ln_base = math.log(base)
        result = 1.0  # First term is 1
        term = 1.0    # Initialize the first term
        
        for n in range(1, terms):
            term *= (exp * ln_base) / n
            result += term
        
        return result

    
    # Calculate rho to the power of eos_Gamma using the series expansion approach
    rho_exp = series_power(rho, eos_Gamma)
    
    # Calculate pressure
    press = eos_kappa * rho_exp
    
    return press


try:
    targets = process_hdf5_to_tuple('58.1', 3)
    target = targets[0]
    rho = 0.1
    eos_Gamma = 2.0
    eos_kappa = 100.
    assert np.allclose(eos_press_from_rho(rho, eos_Gamma, eos_kappa), target)

    target = targets[1]
    rho = 0.2
    eos_Gamma = 3./5.
    eos_kappa = 80
    assert np.allclose(eos_press_from_rho(rho, eos_Gamma, eos_kappa), target)

    target = targets[2]
    rho = 1.1
    eos_Gamma = 1.8
    eos_kappa = 20
    assert np.allclose(eos_press_from_rho(rho, eos_Gamma, eos_kappa), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e