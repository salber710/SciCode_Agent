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
    # Using complex numbers to handle the computation, which is unconventional for real inputs but valid

    press = eos_kappa * (rho ** eos_Gamma).real
    return press



# Background: In a polytropic equation of state, the relationship between pressure (P) and density (ρ) is given by 
# P = κ * ρ^Γ, where κ (eos_kappa) is a constant and Γ (eos_Gamma) is the adiabatic exponent. To find the density 
# given the pressure, we need to rearrange this equation to solve for ρ. This can be done by isolating ρ, resulting 
# in ρ = (P / κ)^(1/Γ). This formula allows us to compute the density from the given pressure using the specified 
# equation of state parameters.

def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: the density corresponding to the given pressure, a float.
    '''
    # Compute the density from the pressure using the inverse of the polytropic equation of state
    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    return rho


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