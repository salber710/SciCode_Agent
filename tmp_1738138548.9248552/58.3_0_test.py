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


def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: the density corresponding to the given pressure, a float.
    '''
    # Using a numerical method to compute the root for the polytropic equation


    def objective(rho):
        return (eos_kappa * rho ** eos_Gamma - press) ** 2

    result = minimize_scalar(objective, bounds=(0, press / eos_kappa * 100), method='bounded')
    return result.x



# Background: In thermodynamics, the specific internal energy (eps) is a measure of the energy stored within a system per unit mass. 
# For a polytropic process, the specific internal energy can be related to the pressure and density using the polytropic equation of state.
# The polytropic equation of state is given by P = κ * ρ^Γ, where P is the pressure, ρ is the density, κ is a constant, and Γ is the adiabatic exponent.
# The specific internal energy can be derived from the first law of thermodynamics and is related to the pressure and density by the relation:
# eps = P / ((Γ - 1) * ρ), where P is the pressure, ρ is the density, and Γ is the adiabatic exponent.

def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    # Calculate density from pressure using the inverse of the polytropic equation
    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    
    # Calculate specific internal energy using the relation eps = P / ((Γ - 1) * ρ)
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