from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy as sp
import scipy.integrate as si



# Background: In astrophysics and thermodynamics, a polytropic process is a thermodynamic process that obeys the equation of state: 
# P = K * rho^Gamma, where P is the pressure, rho is the density, K is a constant known as the polytropic constant, 
# and Gamma is the polytropic index or adiabatic exponent. This equation is used to describe the thermodynamic behavior 
# of a gas under certain conditions. The parameters eos_kappa (K) and eos_Gamma (Gamma) are specific constants 
# that define the particular polytropic process being considered. The task is to compute the pressure given a 
# certain density using these parameters.

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''
    # Compute the pressure using the polytropic equation of state
    press = eos_kappa * (rho ** eos_Gamma)
    
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