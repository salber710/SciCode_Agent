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


def eos_eps_from_density(rho, eos_kappa, eos_Gamma):
    '''This function computes specific internal energy for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    # Calculate specific internal energy using an exponential transformation approach
    if rho > 0:
        eps = eos_kappa * (rho ** eos_Gamma) * (1 - rho ** (1 - eos_Gamma)) / (eos_Gamma - 1)
    else:
        eps = 0  # Handle non-physical or undefined cases for zero or negative density
    return eps



# Background: The Tolman-Oppenheimer-Volkoff (TOV) equation describes the structure of a spherically symmetric body in hydrostatic equilibrium, such as a neutron star. 
# It is derived from the general relativity equations and accounts for the balance between gravitational forces and internal pressure. 
# The TOV equation involves the pressure `press`, mass `mass`, and gravitational potential `phi` as functions of the radial coordinate `r`. 
# At the center of the star, these quantities are constant, and outside the star, they are zero. 
# The integrand `u` for the TOV equation is computed using the density and specific internal energy derived from the pressure using the polytropic equation of state.




def eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.'''
    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    eps = eos_kappa * (rho ** eos_Gamma) * (1 - rho ** (1 - eos_Gamma)) / (eos_Gamma - 1)
    return eps

def rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.'''
    return (press / eos_kappa) ** (1 / eos_Gamma)

def tov_RHS(data, r, eos_Gamma, eos_kappa):
    '''This function computes the integrand of the Tolman-Oppenheimer-Volkoff equation describing a neutron star consisting of a gas described by a polytropic equation of state.
    Inputs:
    data: the state vector, a 3-element tuple consisting of the current values for (`press`, `mass` and `phi`), all floats
    r: the radius at which to evaluate the right-hand-side
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rhs: the integrand of the Tolman-Oppenheimer-Volkoff equation, a 3-element tuple of update terms for (`press`, `mass` and `phi`), all floats. 0 when outside of the star.
    '''
    press, mass, phi = data

    # If pressure is zero or negative, we are outside the star
    if press <= 0:
        return (0.0, 0.0, 0.0)

    # Compute density and specific internal energy from pressure
    rho = rho_from_press(press, eos_Gamma, eos_kappa)
    eps = eps_from_press(press, eos_Gamma, eos_kappa)

    # Gravitational constant G and speed of light c in natural units
    G = 1.0
    c = 1.0

    # Compute the derivatives according to the TOV equations
    dpress_dr = -(G * (rho + press / c**2) * (mass + 4 * np.pi * r**3 * press / c**2)) / (r * (r - 2 * G * mass / c**2))
    dmass_dr = 4 * np.pi * r**2 * rho
    dphi_dr = G * (mass + 4 * np.pi * r**3 * press / c**2) / (r * (r - 2 * G * mass / c**2))

    rhs = (dpress_dr, dmass_dr, dphi_dr)
    return rhs


try:
    targets = process_hdf5_to_tuple('58.4', 3)
    target = targets[0]
    data = (1e35, 0.0, 0.0)  # High pressure, mass = 0, phi = 0 at the origin
    r = 0.0
    eos_Gamma = 2.0
    eos_kappa = 1e-10
    assert np.allclose(tov_RHS(data, r, eos_Gamma, eos_kappa), target)

    target = targets[1]
    data = (10, 20, 1.0)  # Moderate pressure, some mass, some phi inside the star
    r = 1e3
    eos_Gamma = 2.0
    eos_kappa = 1e-3
    assert np.allclose(tov_RHS(data, r, eos_Gamma, eos_kappa), target)

    target = targets[2]
    data = (0.3, 1e3, 1.0)
    r = 20
    eos_Gamma = 2
    eos_kappa = 100
    assert np.allclose(tov_RHS(data, r, eos_Gamma, eos_kappa), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e