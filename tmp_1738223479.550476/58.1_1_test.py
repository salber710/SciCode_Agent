from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy as sp
import scipy.integrate as si



def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''
    # Use a logarithmic transformation for computation
    # Calculate the logarithm of the density
    log_rho = math.log(rho)
    
    # Compute the logarithm of the pressure using transformed variables
    log_press = math.log(eos_kappa) + eos_Gamma * log_rho
    
    # Exponentiate to get the pressure
    press = math.exp(log_press)
    
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