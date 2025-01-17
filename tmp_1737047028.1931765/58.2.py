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
