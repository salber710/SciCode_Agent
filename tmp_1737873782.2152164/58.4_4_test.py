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
    # Calculate density from the pressure using the inverse polytropic equation
    rho = (press / eos_kappa) ** (1 / eos_Gamma)

    # Specific internal energy for a polytropic equation of state can be expressed as
    # eps = press / ((eos_Gamma - 1) * rho)
    # This follows from the thermodynamic relationship between pressure, density,
    # and internal energy in a polytropic fluid
    eps = press / ((eos_Gamma - 1) * rho)

    return eps



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

    # Unpack the state vector
    press, mass, phi = data

    # Check if we are outside the star
    if press <= 0:
        return (0.0, 0.0, 0.0)

    # Compute density from pressure using the polytropic equation of state
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)

    # Compute specific internal energy
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)

    # Calculate the derivatives according to the TOV equations
    dpress_dr = -(G * (rho + press / c**2) * (mass + 4 * np.pi * r**3 * press / c**2)) / (r * (r - 2 * G * mass / c**2))
    dmass_dr = 4 * np.pi * r**2 * rho
    dphi_dr = G * mass / r**2 + 4 * np.pi * G * r * press / c**2

    # Return the derivatives as the integrand
    return (dpress_dr, dmass_dr, dphi_dr)


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