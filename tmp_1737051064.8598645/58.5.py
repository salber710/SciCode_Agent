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


# Background: The Tolman-Oppenheimer-Volkoff (TOV) equation describes the structure of a neutron star in hydrostatic equilibrium. 
# It is derived from the general relativistic equations of stellar structure and accounts for the balance between gravitational forces and pressure gradients within the star. 
# The TOV equation is typically expressed in terms of a set of coupled differential equations for pressure (P), mass (m), and gravitational potential (phi) as functions of the radial coordinate (r).
# At the center of the star, the pressure, mass, and gravitational potential are constant, and the integrand of the TOV equation is zero outside the star.
# The specific internal energy and density are computed using the provided functions `eps_from_press` and `rho_from_press`, which are based on the polytropic equation of state.

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

    # If the pressure is zero or negative, we are outside the star
    if press <= 0:
        return (0.0, 0.0, 0.0)

    # Compute the density from the pressure using the polytropic equation of state
    rho = (press / eos_kappa)**(1 / eos_Gamma)

    # Compute the specific internal energy using the Gamma-law equation of state
    eps = press / ((eos_Gamma - 1) * rho)

    # Compute the derivatives for the TOV equations
    dP_dr = -(rho + press / (eos_Gamma - 1)) * (mass + 4 * np.pi * r**3 * press) / (r * (r - 2 * mass))
    dm_dr = 4 * np.pi * r**2 * rho
    dphi_dr = (mass + 4 * np.pi * r**3 * press) / (r * (r - 2 * mass))

    # Return the right-hand side of the TOV equations
    rhs = (dP_dr, dm_dr, dphi_dr)
    return rhs



# Background: The Tolman-Oppenheimer-Volkoff (TOV) equation describes the structure of a neutron star in hydrostatic equilibrium. 
# It involves solving a set of coupled differential equations for pressure, mass, and gravitational potential as functions of the radial coordinate.
# The gravitational mass of the star is the total mass enclosed within the star's surface, and the gravitational time dilation is related to the gravitational potential at the center.
# The time dilation factor, or lapse function, is given by exp(phi/2), where phi is the gravitational potential.
# To solve the TOV equations, we start from the center of the star with a given central density and integrate outward to the surface where the pressure drops to zero.
# The integration is performed using numerical methods, and the initial conditions are derived from the central density using the polytropic equation of state.



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

    # Initial conditions at the center of the star
    press_c = eos_kappa * rhoc**eos_Gamma  # Central pressure from central density
    mass_c = 0.0  # Mass is zero at the center
    phi_c = 0.0  # Gravitational potential is zero at the center

    # Define the initial state vector
    initial_state = (press_c, mass_c, phi_c)

    # Define the radial grid
    r = np.linspace(0, rmax, npoints)

    # Integrate the TOV equations using scipy's odeint
    def tov_rhs(data, r):
        press, mass, phi = data
        if press <= 0:
            return (0.0, 0.0, 0.0)
        
        # Compute the density from the pressure
        rho = (press / eos_kappa)**(1 / eos_Gamma)
        
        # Compute the specific internal energy
        eps = press / ((eos_Gamma - 1) * rho)
        
        # Compute the derivatives for the TOV equations
        dP_dr = -(rho + press / (eos_Gamma - 1)) * (mass + 4 * np.pi * r**3 * press) / (r * (r - 2 * mass))
        dm_dr = 4 * np.pi * r**2 * rho
        dphi_dr = (mass + 4 * np.pi * r**3 * press) / (r * (r - 2 * mass))
        
        return (dP_dr, dm_dr, dphi_dr)

    # Integrate the TOV equations
    solution = si.odeint(tov_rhs, initial_state, r, tfirst=True)

    # Extract the profiles
    press_profile = solution[:, 0]
    mass_profile = solution[:, 1]
    phi_profile = solution[:, 2]

    # The gravitational mass is the mass at the surface of the star
    star_mass = mass_profile[-1]

    # The lapse function (time dilation) at the center is exp(phi/2)
    star_lapse = np.exp(phi_profile[0] / 2)

    return (star_mass, star_lapse)


from scicode.parse.parse import process_hdf5_to_tuple

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
