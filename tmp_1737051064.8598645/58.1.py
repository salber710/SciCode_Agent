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


from scicode.parse.parse import process_hdf5_to_tuple

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
