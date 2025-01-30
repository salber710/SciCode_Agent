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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('36.2', 3)
target = targets[0]

assert np.allclose(fermi_dirac_integral_half_polylog(-0.1), target)
target = targets[1]

assert np.allclose(fermi_dirac_integral_half_polylog(0), target)
target = targets[2]

assert np.allclose(fermi_dirac_integral_half_polylog(0.05), target)
