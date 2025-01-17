import numpy as np
import scipy as sp
import scipy.integrate as si

# Background: In astrophysics and fluid dynamics, a polytropic equation of state is often used to describe the relationship between pressure and density in a fluid. 
# The equation is given by P = K * rho^Gamma, where P is the pressure, rho is the density, K is a constant (eos_kappa), and Gamma is the adiabatic exponent (eos_Gamma). 
# This equation is useful in modeling the behavior of gases under various conditions, such as in stars or planetary atmospheres. 
# The adiabatic exponent, Gamma, determines how compressible the fluid is, with higher values indicating less compressibility.

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''
    # Calculate the pressure using the polytropic equation of state
    press = eos_kappa * rho**eos_Gamma
    return press


# Background: In astrophysics and fluid dynamics, the polytropic equation of state is used to relate pressure and density in a fluid. 
# The equation is given by P = K * rho^Gamma, where P is the pressure, rho is the density, K is a constant (eos_kappa), and Gamma is the adiabatic exponent (eos_Gamma). 
# To find the density given the pressure, we need to rearrange this equation to solve for rho. 
# This involves taking the inverse of the power of Gamma and dividing by K, resulting in rho = (P / K)^(1/Gamma). 
# This rearrangement allows us to compute the density from a known pressure using the same parameters of the equation of state.

def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: density corresponding to the given pressure, a float.
    '''
    # Calculate the density using the rearranged polytropic equation of state
    rho = (press / eos_kappa)**(1 / eos_Gamma)
    return rho



# Background: In astrophysics and fluid dynamics, the specific internal energy (ε) is an important quantity that represents the energy per unit mass contained within a system. 
# For a polytropic equation of state combined with the Gamma-law equation of state, the specific internal energy can be derived from the pressure and density relationship. 
# The Gamma-law equation of state relates the pressure (P), density (ρ), and specific internal energy (ε) through the relation P = (Γ - 1) * ρ * ε, where Γ is the adiabatic exponent. 
# By rearranging this equation, we can solve for the specific internal energy as ε = P / ((Γ - 1) * ρ). 
# This allows us to compute the specific internal energy given the pressure, density, and the adiabatic exponent.

def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    # Calculate the density using the rearranged polytropic equation of state
    rho = (press / eos_kappa)**(1 / eos_Gamma)
    
    # Calculate the specific internal energy using the Gamma-law equation of state
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
