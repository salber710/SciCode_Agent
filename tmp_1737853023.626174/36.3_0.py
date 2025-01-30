import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton

# Background: 
# The generation of electron-hole pairs in a semiconductor due to incident light is a fundamental concept in optoelectronics. 
# When light with a certain wavelength (λ) is incident on a semiconductor, it can be absorbed, generating electron-hole pairs. 
# The rate of generation of these carriers depends on the optical power (P), the area over which the light is incident (A), 
# the absorption coefficient (α), and the depth (x) into the material. The absorption coefficient α indicates how quickly 
# the light intensity decreases as it penetrates the material. The generation rate of carriers is also influenced by the 
# lifetime (τ) of the carriers, which is the average time an electron or hole exists before recombination. 
# The energy of the incident photons is given by E = h*c/λ, where h is Planck's constant and c is the speed of light. 
# The number of photons incident per second is P/E, and the number of photons absorbed per unit volume per second at a 
# depth x is given by (α * P * exp(-α * x)) / (A * E). The generated carrier density n(x) is then the product of this 
# generation rate and the carrier lifetime τ.


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
    h = 6.626e-34  # Planck's constant in J·s

    # Convert units
    A_m2 = A * 1e-12  # Convert area from μm^2 to m^2
    lambda_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    alpha_m = alpha * 1e2  # Convert absorption coefficient from cm^-1 to m^-1
    x_m = x * 1e-6  # Convert depth from μm to m

    # Check for physically impossible or undefined conditions
    if lambda_i <= 0:
        raise ValueError("Wavelength must be greater than zero.")
    if P < 0 or alpha < 0 or tau < 0 or x < 0:
        return 0

    # Energy of a single photon
    E_photon = h * c / lambda_m  # in Joules

    # Photon flux (number of photons per second per unit area)
    photon_flux = P / E_photon  # in photons/s

    # Generation rate of electron-hole pairs per unit volume at depth x
    G_x = (alpha_m * photon_flux * np.exp(-alpha_m * x_m)) / A_m2  # in m^-3 s^-1

    # Convert generation rate to cm^-3 s^-1
    G_x_cm3 = G_x * 1e-6

    # Generated carrier density
    dN = G_x_cm3 * tau  # in cm^-3

    return dN


# Background: The Fermi-Dirac distribution describes the occupancy of electron energy states in a semiconductor at thermal equilibrium.
# The electron density in the conduction band can be calculated by integrating the product of the density of states and the Fermi-Dirac
# distribution over energy. For semiconductors like GaAs, the effective mass approximation is used, where the effective mass of electrons
# is a fraction of the free electron mass. The Fermi-Dirac integral of order 1/2 is often used to compute the electron density in the
# conduction band. The polylogarithm function can be used to evaluate this integral. The thermal voltage at room temperature is used to
# convert energy levels from electron volts to joules.




def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    # Constants
    m0 = 9.109e-31  # Electron rest mass in kg
    me = 0.067 * m0  # Effective mass of electron in GaAs
    h = 6.626e-34  # Planck's constant in J·s
    q = 1.602e-19  # Electron charge in C
    kT = 0.0259  # Thermal voltage at room temperature in eV

    # Convert Ef from eV to Joules
    Ef_J = Ef * q

    # Density of states effective mass
    def density_of_states(E):
        return (2 * me**1.5 / (2 * np.pi**2 * h**3)) * np.sqrt(E)

    # Fermi-Dirac distribution
    def fermi_dirac(E):
        return expit((Ef_J - E) / (kT * q))

    # Integrand for the Fermi-Dirac integral
    def integrand(E):
        return density_of_states(E) * fermi_dirac(E)

    # Perform the integral from 0 to infinity
    integral, _ = quad(integrand, 0, np.inf)

    # Convert the result to cm^-3
    n = integral * 1e-6

    return n



# Background: The Fermi level in a semiconductor is a crucial parameter that determines the distribution of electrons
# in the conduction band. The Fermi-Dirac integral of order 1/2 is used to relate the electron density to the Fermi level.
# In this step, we aim to find the Fermi level as a function of the electron density using the inverse of the Fermi-Dirac
# integral. The Newton-Raphson method is a numerical technique used to find roots of real-valued functions, which is
# suitable for solving the implicit equation relating the Fermi level and electron density. If the electron density is not
# provided, it can be calculated using the generation function, which computes the excess electron distribution based on
# the incident optical power, beam area, wavelength, absorption coefficient, carrier lifetime, and depth.




def generation(P, A, lambda_i, alpha, tau, x):
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J·s

    A_m2 = A * 1e-12  # Convert area from μm^2 to m^2
    lambda_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    alpha_m = alpha * 1e2  # Convert absorption coefficient from cm^-1 to m^-1
    x_m = x * 1e-6  # Convert depth from μm to m

    if lambda_i <= 0:
        raise ValueError("Wavelength must be greater than zero.")
    if P < 0 or alpha < 0 or tau < 0 or x < 0:
        return 0

    E_photon = h * c / lambda_m  # in Joules
    photon_flux = P / E_photon  # in photons/s
    G_x = (alpha_m * photon_flux * np.exp(-alpha_m * x_m)) / A_m2  # in m^-3 s^-1
    G_x_cm3 = G_x * 1e-6
    dN = G_x_cm3 * tau  # in cm^-3

    return dN

def inverse_fermi_dirac_integral_half_polylog_newton(P, A, lambda_i, alpha, tau, x, n=None):
    m0 = 9.109e-31  # Electron rest mass in kg
    me = 0.067 * m0  # Effective mass of electron in GaAs
    h = 6.626e-34  # Planck's constant in J·s
    q = 1.602e-19  # Electron charge in C
    kT = 0.0259  # Thermal voltage at room temperature in eV

    if n is None:
        n = generation(P, A, lambda_i, alpha, tau, x)

    def fermi_dirac_integral_half(Ef):
        Ef_J = Ef * q

        def density_of_states(E):
            return (2 * me**1.5 / (2 * np.pi**2 * h**3)) * np.sqrt(E)

        def fermi_dirac(E):
            return 1 / (1 + np.exp((E - Ef_J) / (kT * q)))

        def integrand(E):
            return density_of_states(E) * fermi_dirac(E)

        integral, _ = quad(integrand, 0, np.inf)
        return integral * 1e-6

    def root_function(Ef):
        return fermi_dirac_integral_half(Ef) - n

    Ef_initial_guess = 0.1  # Initial guess for the Fermi level in eV
    Ef = newton(root_function, Ef_initial_guess)

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
