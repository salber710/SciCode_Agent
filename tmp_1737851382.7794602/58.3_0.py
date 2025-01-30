import numpy as np
import scipy as sp
import scipy.integrate as si

# Background: In astrophysics and fluid dynamics, a polytropic equation of state is often used to describe the relationship between pressure and density in a fluid. 
# The polytropic equation of state is given by the formula P = K * rho^Gamma, where P is the pressure, rho is the density, K is a constant (eos_kappa), 
# and Gamma (eos_Gamma) is the adiabatic exponent. This equation is useful in modeling the behavior of gases under various conditions, 
# such as in stars or planetary atmospheres, where the pressure and density are related in a non-linear manner.

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''
    if rho < 0:
        raise ValueError("Density rho cannot be negative.")
    press = eos_kappa * rho**eos_Gamma
    return press


# Background: In astrophysics and fluid dynamics, the polytropic equation of state is used to relate pressure and density in a fluid. 
# The equation is given by P = K * rho^Gamma, where P is the pressure, rho is the density, K is a constant (eos_kappa), 
# and Gamma (eos_Gamma) is the adiabatic exponent. To find the density given the pressure, we need to rearrange this equation 
# to solve for rho. This can be done by taking the inverse of the equation: rho = (P / K)^(1/Gamma). 
# This rearrangement allows us to compute the density from a given pressure using the same parameters of the equation of state.

def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: density corresponding to the given pressure, a float.
    '''
    if press < 0:
        raise ValueError("Pressure press cannot be negative.")
    if eos_Gamma <= 0:
        raise ValueError("Adiabatic exponent eos_Gamma must be positive.")
    if eos_kappa <= 0:
        raise ValueError("Coefficient eos_kappa must be positive.")
    
    rho = (press / eos_kappa)**(1 / eos_Gamma)
    return rho



# Background: In astrophysics and fluid dynamics, the specific internal energy is an important quantity that describes the energy per unit mass contained within a system. 
# For a polytropic equation of state, the specific internal energy can be related to the pressure and density. 
# The specific internal energy, ε, can be derived from the first law of thermodynamics and is given by the formula:
# ε = P / ((Γ - 1) * ρ), where P is the pressure, ρ is the density, and Γ (eos_Gamma) is the adiabatic exponent. 
# This formula arises from the assumption of a polytropic process, where the internal energy is related to the work done by/on the system.

def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    if press < 0:
        raise ValueError("Pressure press cannot be negative.")
    if eos_Gamma <= 1:
        raise ValueError("Adiabatic exponent eos_Gamma must be greater than 1 for a valid specific internal energy calculation.")
    if eos_kappa <= 0:
        raise ValueError("Coefficient eos_kappa must be positive.")
    
    # Calculate density from pressure using the inverse of the polytropic equation of state
    rho = (press / eos_kappa)**(1 / eos_Gamma)
    
    # Calculate specific internal energy using the derived formula
    eps = press / ((eos_Gamma - 1) * rho)
    
    return eps

from scicode.parse.parse import process_hdf5_to_tuple
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
