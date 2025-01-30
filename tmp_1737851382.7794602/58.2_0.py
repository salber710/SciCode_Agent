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
    rho = (press / eos_kappa)**(1 / eos_Gamma)
    return rho

from scicode.parse.parse import process_hdf5_to_tuple
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
