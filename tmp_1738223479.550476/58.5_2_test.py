from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density 
    using a continued fraction approximation for exponentiation.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''

    # Continued fraction approximation for power function
    def continued_fraction_power(base, exp, iterations=10):
        # Initialize continued fraction components
        a = [exp] * iterations
        b = [1] * iterations
        b[0] = base  # Start with base as the first b term

        # Compute continued fraction
        result = a[-1]
        for i in range(iterations - 2, -1, -1):
            result = a[i] + b[i] / result

        return base ** exp / result

    # Calculate rho to the power of eos_Gamma using continued fraction approximation
    rho_exp = continued_fraction_power(rho, eos_Gamma)

    # Calculate pressure
    press = eos_kappa * rho_exp

    return press


def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: the density corresponding to the given pressure, a float
    '''

    # Using a series expansion method to approximate the density
    # We'll use a binomial series expansion approach to find the density.

    # Starting with an initial guess
    rho_initial = (press / eos_kappa) ** (1 / eos_Gamma)

    # Using a small perturbation to refine the estimate via series expansion
    delta = 1e-6  # Small perturbation for series expansion
    rho_estimate = rho_initial

    # Calculate the pressure for the initial estimate
    press_estimate = eos_kappa * rho_estimate ** eos_Gamma
    
    # Calculate the first derivative of the pressure
    dP_drho = eos_kappa * eos_Gamma * rho_estimate ** (eos_Gamma - 1)

    # Use a binomial expansion to adjust the estimate
    term = (press - press_estimate) / dP_drho
    rho_estimate += term

    # Calculate higher-order corrections
    correction_factor = 0.5  # Arbitrary correction factor for demonstration
    for _ in range(3):  # Limit to a few terms for simplicity
        term = correction_factor * (press - press_estimate) ** 2 / (dP_drho ** 2)
        rho_estimate += term
        press_estimate = eos_kappa * rho_estimate ** eos_Gamma
        dP_drho = eos_kappa * eos_Gamma * rho_estimate ** (eos_Gamma - 1)

    return rho_estimate


def eos_compute_eps_via_entropy_density(rho, eos_kappa, eos_Gamma):
    '''This function computes specific internal energy for a polytropic equation of state given the density.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    
    # Define an entropy density based on a novel interpretation of the polytropic relation
    entropy_density = eos_kappa * (rho ** eos_Gamma) / (eos_Gamma + 1)
    
    # Compute specific internal energy using this entropy density
    # Consider the influence of the entropy density on specific internal energy
    eps = entropy_density * (eos_Gamma - 1) / (rho ** (eos_Gamma - 2))
    
    return eps


def tov_RHS(data, r, eos_Gamma, eos_kappa):
    '''This function computes the integrand of the Tolman-Oppenheimer-Volkoff equation describing a neutron star consisting of a gas described by a polytropic equation of state.
    Inputs:
    data: the state vector, a 3-element tuple consisting of the current values for (`press`, `mass`, and `phi`), all floats
    r: the radius at which to evaluate the right-hand-side
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rhs: the integrand of the Tolman-Oppenheimer-Volkoff equation, a 3-element tuple of update terms for (`press`, `mass`, and `phi`), all floats. 0 when outside of the star.
    '''

    press, mass, phi = data

    # Check if outside the star
    if press <= 0 or r <= 0:
        return (0.0, 0.0, 0.0)

    # Calculate density and specific internal energy using the provided eos functions
    rho = rho_from_press(press, eos_Gamma, eos_kappa)
    eps = eps_from_press(press, eos_Gamma, eos_kappa)

    # Use a unique approach to calculate the gravitational factor
    grav_term = (mass + 4.0 * 3.141592653589793 * r**3 * press)
    radius_term = max(r * (r - 2 * mass), 1e-10)
    
    # Use scaling factors for calculations
    scale_factor = 1e-10
    dP_dr = -(rho + press + eps) * grav_term / radius_term * scale_factor
    dM_dr = (4.0 * 3.141592653589793 * r**2 * rho) * scale_factor
    dPhi_dr = grav_term / radius_term * scale_factor

    return (dP_dr, dM_dr, dPhi_dr)





def press_from_rho(rho, eos_Gamma, eos_kappa):
    """Compute pressure from density using a polytropic equation of state."""
    return eos_kappa * rho ** eos_Gamma

def tov_rhs(r, y, eos_Gamma, eos_kappa):
    """Compute derivatives for the TOV equations."""
    press, mass, phi = y

    if press <= 0:
        return [0.0, 0.0, 0.0]

    rho = (press / eos_kappa) ** (1 / eos_Gamma)
    eps = eos_kappa * rho ** eos_Gamma / (eos_Gamma - 1)

    grav_term = (mass + 4 * np.pi * r**3 * press)
    radius_term = r * (r - 2 * mass)

    dP_dr = -(rho + press + eps) * grav_term / (radius_term + 1e-10)
    dM_dr = 4 * np.pi * r**2 * rho
    dPhi_dr = grav_term / (radius_term + 1e-10)

    return [dP_dr, dM_dr, dPhi_dr]

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

    # Initial conditions
    r_initial = 1e-9  # small radius to avoid singularity
    press_initial = press_from_rho(rhoc, eos_Gamma, eos_kappa)
    mass_initial = 0.0
    phi_initial = 0.0

    initial_conditions = [press_initial, mass_initial, phi_initial]

    # Radial points
    r_points = np.linspace(r_initial, rmax, npoints)

    # Integrate the TOV equations
    solution = solve_ivp(tov_rhs, [r_initial, rmax], initial_conditions, args=(eos_Gamma, eos_kappa), t_eval=r_points, method='RK23', rtol=1e-6, atol=1e-9)

    pressures = solution.y[0]
    masses = solution.y[1]
    potentials = solution.y[2]

    # Find the surface of the star
    for i, p in enumerate(pressures):
        if p <= 0:
            break

    star_mass = masses[i-1]
    star_phi_surface = potentials[i-1]

    # Compute the gravitational time dilation at the center
    star_lapse = np.exp(star_phi_surface - phi_initial)

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