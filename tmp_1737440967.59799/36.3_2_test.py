import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton

# Background: 
# The generation of electron-hole pairs in a semiconductor due to incident light is a fundamental process in optoelectronics.
# The number of generated carriers can be determined by the optical power absorbed per unit volume.
# The incident optical power (P) is distributed over the beam area (A), giving the power density.
# The energy of each photon is given by E_photon = hc/λ, where h is Planck's constant and c is the speed of light.
# The rate of electron-hole pair generation per unit volume, also known as the generation rate G(x), at a depth x is given by:
# G(x) = (α * P_absorbed(x)) / (A * E_photon), where P_absorbed(x) = P * e^(-αx) is the power absorbed at depth x.
# To find the excess electron density n(x), we consider the balance between generation and recombination,
# leading to the steady-state solution: n(x) = G(x) * τ, where τ is the carrier lifetime.
# Note: Depth x should be converted from micrometers to centimeters as absorption coefficient α is in cm^-1.

def generation(P, A, lambda_i, alpha, tau, x):
    '''This function computes the excess electron distribution.
    Input:
    P (float): incident optical power in W
    A (float): beam area in μm^2
    lambda_i (float): incident wavelength in nm
    alpha (float): absorption coefficient in cm^-1
    tau (float): lifetime of excess carriers in s
    x (float): depth variable in μm
    Output:
    dN (float): generated carrier density in cm^-3
    '''
    # Constants
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s
    
    # Convert wavelength from nm to m
    lambda_m = lambda_i * 1e-9
    
    # Convert beam area from μm^2 to cm^2
    A_cm2 = A * 1e-8
    
    # Convert depth x from μm to cm
    x_cm = x * 1e-4
    
    # Calculate the energy of a photon
    E_photon = (h * c) / lambda_m  # Energy in Joules
    
    # Calculate the absorbed power per unit volume at depth x
    P_absorbed_x = P * np.exp(-alpha * x_cm)
    
    # Calculate the generation rate G(x) in cm^-3 s^-1
    G_x = (alpha * P_absorbed_x) / (A_cm2 * E_photon)
    
    # Calculate the excess electron density n(x) in cm^-3
    dN = G_x * tau
    
    return dN


# Background: In semiconductors, the electron density in the conduction band at thermal equilibrium can be calculated using the Fermi-Dirac distribution. 
# The Fermi-Dirac integral of order 1/2 is used to model this behavior and is related to the density of states in the conduction band.
# For a material like GaAs, the effective density of states in the conduction band Nc can be computed using:
# Nc = 2 * ((2 * pi * m_e * kT) / (h^2))^(3/2), where m_e is the effective electron mass, k is Boltzmann's constant, and T is temperature.
# The thermal voltage Vt at room temperature is approximately 0.0259 V and is used to relate energy to temperature.
# The Fermi-Dirac integral of order 1/2 can be expressed using the polylogarithm function, specifically the polylogarithm of order 3/2.
# The integral is performed over the energy range, and the result is multiplied by the density of states to find the electron density n.





def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    # Constants
    m0 = 9.109e-31  # electron rest mass in kg
    me = 0.067 * m0  # effective mass of electron in GaAs
    h = 6.626e-34  # Planck's constant in J*s
    q = 1.602e-19  # electron charge in C
    k = 1.380649e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in Kelvin
    Vt = 0.0259  # Thermal voltage at room temperature in V

    # Calculate effective density of states Nc in cm^-3
    Nc = 2 * ((2 * np.pi * me * k * T) / (h ** 2)) ** (3/2) / 1e6  # convert from m^-3 to cm^-3

    # Calculate the Fermi-Dirac integral of order 1/2 using the polylog function
    F_half = polylog(1.5, np.exp(Ef / Vt))

    # Calculate the electron density n
    n = Nc * F_half

    return n



# Background: In order to determine the Fermi level as a function of electron density, we need to inverse the 
# Fermi-Dirac integral of order 1/2, which relates the electron density to the Fermi level. This involves solving 
# for the Fermi level given an electron density. The Newton-Raphson method is a numerical technique used to find 
# approximate roots of real-valued functions. Here, we will use it to solve for the Fermi level such that the 
# calculated electron density matches the given (or calculated) electron density. If the electron density is not 
# provided, it can be calculated using the generation function for excess carriers. The effective electron mass 
# in GaAs and constants like Planck's constant, electron charge, Boltzmann's constant, and temperature are used 
# to compute the effective density of states and the polylogarithm function is used to compute the Fermi-Dirac 
# integral.





def generation(P, A, lambda_i, alpha, tau, x):
    '''This function computes the excess electron distribution.'''
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J*s
    
    lambda_m = lambda_i * 1e-9
    A_cm2 = A * 1e-8
    x_cm = x * 1e-4
    
    E_photon = (h * c) / lambda_m
    
    P_absorbed_x = P * np.exp(-alpha * x_cm)
    
    G_x = (alpha * P_absorbed_x) / (A_cm2 * E_photon)
    
    dN = G_x * tau
    
    return dN

def inverse_fermi_dirac_integral_half_polylog_newton(P, A, lambda_i, alpha, tau, x, n=None):
    '''This function uses the Newton-Raphson method to find the root of an implicit function.'''
    m0 = 9.109e-31  # electron rest mass in kg
    me = 0.067 * m0  # effective mass of electron in GaAs
    h = 6.626e-34  # Planck's constant in J*s
    q = 1.602e-19  # electron charge in C
    k = 1.380649e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in Kelvin
    Vt = 0.0259  # Thermal voltage at room temperature in V

    Nc = 2 * ((2 * np.pi * me * k * T) / (h ** 2)) ** (3/2) / 1e6  # convert from m^-3 to cm^-3

    if n is None:
        n = generation(P, A, lambda_i, alpha, tau, x)

    def f(Ef):
        return Nc * polylog(1.5, np.exp(Ef / Vt)) - n

    Ef = newton(f, 0.1)  # Initial guess for Ef

    return Ef

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('36.3', 4)
target = targets[0]

m_eff = 0.067 * 9.109e-31  # Effective mass of electrons in GaAs (kg)
h = 6.626e-34  # Planck's constant (J*s)
kT = .0259
q = 1.602e-19
N_c = 2 * ((2 * np.pi * m_eff * kT*q) / (h**2))**(3/2) *100**-3  # Effective density of states in the conduction band (cm^-3)
Ef = inverse_fermi_dirac_integral_half_polylog_newton(1e-3, 50, 519, 1e4, 1e-9, 1,N_c)
assert (np.isclose(Ef,0,atol=0.02)) == target
target = targets[1]

assert np.allclose(inverse_fermi_dirac_integral_half_polylog_newton(1e-3, 50, 519, 1e4, 1e-9, 0.4), target)
target = targets[2]

assert np.allclose(inverse_fermi_dirac_integral_half_polylog_newton(10e-3, 50, 519, 1e4, 1e-9, 0.4), target)
target = targets[3]

assert np.allclose(inverse_fermi_dirac_integral_half_polylog_newton(100e-3, 50, 519, 1e4, 1e-9, 1), target)
