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
    # The pressure is calculated using the polytropic equation of state: P = K * rho^Gamma
    # where K is the eos_kappa and Gamma is the eos_Gamma
    press = eos_kappa * rho ** eos_Gamma
    return press


def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: the density corresponding to the given pressure, a float.
    '''
    # Rearrange the polytropic equation of state P = K * rho^Gamma to solve for rho
    # rho = (P / K)^(1/Gamma)
    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    return rho



def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    
    # Specific internal energy can be related to pressure and density for a polytropic equation of state.
    # Using the relation: eps = P / ((Gamma - 1) * rho), where rho is the density
    # First, we need to compute density rho using the inverse of the polytropic equation: rho = (P / eos_kappa)^(1/eos_Gamma)
    
    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    
    # Now compute specific internal energy
    eps = press / ((eos_Gamma - 1) * rho)
    
    return eps


try:
    targets = process_hdf5_to_tuple('58.3', 3)
    target = targets[0]
    press = 10
    eos_Gamma = 15
    eos_kappa = 20
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

    target = targets[1]
    press = 10000
    eos_Gamma = 3./5.
    eos_kappa = 80
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

    target = targets[2]
    press = 100
    eos_Gamma = 2.
    eos_kappa = 100.
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e