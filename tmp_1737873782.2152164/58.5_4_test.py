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
    '''This function computes the integrand of the Tolman-Oppenheimer-Volkoff equation describing a neutron star
    consisting of a gas described by a polytropic equation of state.
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

    # If pressure is zero or negative, we are outside the star; return zero for all quantities
    if press <= 0:
        return (0, 0, 0)

    # Compute density using the provided function
    rho = eos_rho_from_press(press, eos_Gamma, eos_kappa)

    # Compute specific internal energy using the provided function
    eps = eos_eps_from_press(press, eos_Gamma, eos_kappa)

    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    c = 299792458    # speed of light in m/s

    # Compute the derivatives using the TOV equations

    # dP/dr
    dP_dr = -G * (rho + eps/c**2) * (mass + 4 * np.pi * r**3 * press/c**2) / (r**2 * (1 - 2 * G * mass / (c**2 * r)))

    # dm/dr
    dm_dr = 4 * np.pi * r**2 * rho

    # dphi/dr
    dphi_dr = G * (mass + 4 * np.pi * r**3 * press/c**2) / (c**2 * r * (1 - 2 * G * mass / (c**2 * r)))

    # Return the derivatives as a tuple
    return (dP_dr, dm_dr, dphi_dr)





def tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax):
    '''This function computes gravitational time dilation at the center of the neutron star described by a polytropic equation of state as well as the star's mass.
    Inputs
    rhoc: float, the density at the center of the star, in units where G=c=Msun=1.
    Gamma: float, adiabatic exponent of the equation of state
    kappa: float, coefficient of the equation of state
    npoints: int, number of integration points to use
    rmax: float, maximum radius to which to integrate solution to, must include the whole star
    Outputs
    mass: float, gravitational mass of neutron star, in units where G=c=Msun=1
    lapse: float, gravitational time dilation at center of neutron star
    '''

    # Define the initial conditions of the star at the center
    press_c = eos_press_from_rho(rhoc, eos_Gamma, eos_kappa)
    mass_c = 0.0
    phi_c = 0.0
    init_conditions = (press_c, mass_c, phi_c)

    # Define the radial points for integration
    r = np.linspace(1e-5, rmax, npoints)

    # Use scipy's odeint to integrate the TOV equations
    result = si.odeint(tov_RHS, init_conditions, r, args=(eos_Gamma, eos_kappa))

    # Extract the profiles
    pressures = result[:, 0]
    masses = result[:, 1]
    potentials = result[:, 2]

    # Find the surface of the star (where pressure effectively goes to zero)
    surface_index = np.where(pressures <= 0)[0][0] if np.any(pressures <= 0) else -1

    # Gravitational mass is the mass at the surface
    star_mass = masses[surface_index]

    # Gravitational time dilation at the center of the star
    star_lapse = np.exp(potentials[0] - potentials[surface_index])

    return (star_mass, star_lapse)

# Helper functions (assuming they are defined elsewhere according to previous steps)
# def eos_press_from_rho(rho, eos_Gamma, eos_kappa): ...
# def tov_RHS(data, r, eos_Gamma, eos_kappa): ...


try:
    targets = process_hdf5_to_tuple('58.5', 4)
    target = targets[0]
    rhoc = 0.3
    eos_Gamma = 2.1
    eos_kappa = 30
    npoints = 200
    rmax = 100000.
    assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)

    target = targets[1]
    rhoc = 2e-5
    eos_Gamma = 1.8
    eos_kappa = 20
    npoints = 2000
    rmax = 100.
    assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)

    target = targets[2]
    rhoc = 1.28e-3
    eos_Gamma = 5./3.
    eos_kappa = 80.
    npoints = 200000
    rmax = 100.
    assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)

    target = targets[3]
    rhoc = 1.28e-3
    # equation of state
    eos_Gamma = 2.0
    eos_kappa = 100.
    # grid for integration
    rmax = 100.
    npoints = 200000
    assert np.allclose(tov(rhoc, eos_Gamma, eos_kappa, npoints, rmax), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e