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


def tov_RHS(data, r, eos_Gamma, eos_kappa):
    press, mass, phi = data
    if press <= 0:
        return (0.0, 0.0, 0.0)

    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    eps = eos_kappa * rho ** eos_Gamma / (eos_Gamma - 1)

    G = 1.0  # Use natural units where G = 1
    c = 1.0  # Use natural units where c = 1

    # Using a different approach to handle the singularity at r = 0
    if r < 1e-10:
        dpress_dr = 0.0
        dmass_dr = 0.0
        dphi_dr = 0.0
    else:
        dpress_dr = -(G * (rho + eps / c**2 + press / c**2) * (mass + 4 * np.pi * r**3 * press / c**2)) / (r * (r - 2 * G * mass / c**2))
        dmass_dr = 4 * np.pi * r**2 * rho
        dphi_dr = G * (mass + 4 * np.pi * r**3 * press / c**2) / (r * (r - 2 * G * mass / c**2))

    return (dpress_dr, dmass_dr, dphi_dr)



# Background: The Tolman-Oppenheimer-Volkoff (TOV) equations describe the structure of a spherically symmetric body in hydrostatic equilibrium, such as a neutron star, in general relativity. The TOV equations are derived from the Einstein field equations and account for the balance between gravitational forces and pressure gradients within the star. The gravitational mass of the star is determined by integrating the mass density over the star's volume, while the gravitational time dilation at the center is related to the gravitational potential. The time dilation factor, or lapse function, is a measure of how much time slows down in a gravitational field compared to a distant observer. In this function, we will integrate the TOV equations from the center of the star to its surface, using a polytropic equation of state to relate pressure and density.




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

    # Define the initial conditions at the center of the star
    press_c = eos_kappa * rhoc ** eos_Gamma
    mass_c = 0.0
    phi_c = 0.0

    # Define the radial grid
    r = np.linspace(0, rmax, npoints)

    # Define the function to compute the derivatives (RHS of TOV equations)
    def tov_RHS(data, r):
        press, mass, phi = data
        if press <= 0:
            return (0.0, 0.0, 0.0)

        rho = (press / eos_kappa) ** (1 / eos_Gamma)
        eps = eos_kappa * rho ** eos_Gamma / (eos_Gamma - 1)

        G = 1.0  # Use natural units where G = 1
        c = 1.0  # Use natural units where c = 1

        if r < 1e-10:
            dpress_dr = 0.0
            dmass_dr = 0.0
            dphi_dr = 0.0
        else:
            dpress_dr = -(G * (rho + eps / c**2 + press / c**2) * (mass + 4 * np.pi * r**3 * press / c**2)) / (r * (r - 2 * G * mass / c**2))
            dmass_dr = 4 * np.pi * r**2 * rho
            dphi_dr = G * (mass + 4 * np.pi * r**3 * press / c**2) / (r * (r - 2 * G * mass / c**2))

        return (dpress_dr, dmass_dr, dphi_dr)

    # Integrate the TOV equations using scipy's odeint
    u0 = [press_c, mass_c, phi_c]
    sol = si.odeint(tov_RHS, u0, r, tfirst=True)

    # Extract the mass and potential at the surface of the star
    star_mass = sol[-1, 1]
    star_phi = sol[-1, 2]

    # Compute the gravitational time dilation (lapse function) at the center
    star_lapse = np.exp(star_phi)

    return (star_mass, star_lapse)


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