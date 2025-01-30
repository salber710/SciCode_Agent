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
    if eos_Gamma <= 0:
        raise ValueError("Adiabatic exponent eos_Gamma must be positive.")
    if eos_kappa <= 0:
        raise ValueError("Coefficient eos_kappa must be positive.")
    
    rho = (press / eos_kappa)**(1 / eos_Gamma)
    return rho


# Background: In astrophysics and fluid dynamics, the specific internal energy is an important quantity that describes the energy per unit mass contained within a system. 
# For a polytropic equation of state, the specific internal energy can be related to the pressure and density. 
# The specific internal energy, ε, can be derived from the first law of thermodynamics and is given by the formula:
# ε = P / ((Γ - 1) * ρ), where P is the pressure, ρ is the density, and Γ (eos_Gamma) is the adiabatic exponent. 
# This formula arises from the assumption of a polytropic process, where the internal energy is related to the work done by/on the system.

def eos_eps_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes specific internal energy for a polytropic equation of state given the pressure.
    Inputs:
    press: the pressure, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    eps: the specific internal energy, a float.
    '''
    if press < 0:
        raise ValueError("Pressure press cannot be negative.")
    if eos_Gamma <= 1:
        raise ValueError("Adiabatic exponent eos_Gamma must be greater than 1 for a valid specific internal energy calculation.")
    if eos_kappa <= 0:
        raise ValueError("Coefficient eos_kappa must be positive.")
    
    # Calculate density from pressure using the inverse of the polytropic equation of state
    rho = (press / eos_kappa)**(1 / eos_Gamma) if press > 0 else 0
    
    # Calculate specific internal energy using the derived formula
    eps = press / ((eos_Gamma - 1) * rho) if rho > 0 else 0
    
    return eps



# Background: The Tolman-Oppenheimer-Volkoff (TOV) equation describes the structure of a spherically symmetric body in hydrostatic equilibrium, such as a neutron star. 
# It is derived from the general relativity equations and accounts for the balance between gravitational forces and internal pressure. 
# The TOV equation is expressed in terms of three main quantities: pressure (P), mass (m), and gravitational potential (phi). 
# The integrand of the TOV equation describes how these quantities change with respect to the radius (r) inside the star. 
# At the center of the star, these quantities are constant, and outside the star, they are zero. 
# The functions `eps_from_press` and `rho_from_press` are used to compute specific internal energy and density from pressure, respectively, 
# which are necessary for calculating the integrand of the TOV equation.

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

    # If the pressure is zero or negative, we are outside the star, return zero for all quantities
    if press <= 0:
        return (0.0, 0.0, 0.0)

    # Calculate density from pressure using the polytropic equation of state
    rho = (press / eos_kappa)**(1 / eos_Gamma)

    # Calculate specific internal energy using the derived formula
    eps = press / ((eos_Gamma - 1) * rho)

    # Calculate the derivatives according to the TOV equations
    dP_dr = -(rho + press) * (mass + 4 * np.pi * r**3 * press) / (r * (r - 2 * mass))
    dM_dr = 4 * np.pi * r**2 * rho
    dPhi_dr = (mass + 4 * np.pi * r**3 * press) / (r * (r - 2 * mass))

    # Return the right-hand side of the TOV equations
    rhs = (dP_dr, dM_dr, dPhi_dr)
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
