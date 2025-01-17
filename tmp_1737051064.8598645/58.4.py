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


from scicode.parse.parse import process_hdf5_to_tuple

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
