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
# depth x is given by (P/E) * (α/A) * exp(-α*x). The generated carrier density n(x) is then this rate multiplied by the 
# carrier lifetime τ.

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
    lambda_m = lambda_i * 1e-9  # Convert wavelength from nm to m
    A_m2 = A * 1e-12  # Convert area from μm^2 to m^2
    alpha_m = alpha * 1e2  # Convert absorption coefficient from cm^-1 to m^-1
    x_m = x * 1e-6  # Convert depth from μm to m

    # Calculate photon energy
    E_photon = h * c / lambda_m  # Energy of a single photon in Joules

    # Calculate photon flux (number of photons per second per unit area)
    photon_flux = P / E_photon  # Total number of photons per second

    # Calculate generation rate at depth x
    G_x = (photon_flux / A_m2) * alpha_m * np.exp(-alpha_m * x_m)  # Generation rate in m^-3·s^-1

    # Calculate generated carrier density
    dN = G_x * tau  # Carrier density in m^-3

    # Convert carrier density to cm^-3
    dN_cm3 = dN * 1e-6

    return dN_cm3



# Background: 
# The Fermi-Dirac distribution describes the probability of occupancy of energy states by electrons in a semiconductor.
# The Fermi-Dirac integral of order 1/2 is used to calculate the electron density in the conduction band of a semiconductor.
# The effective density of states in the conduction band (Nc) for a semiconductor like GaAs can be calculated using the 
# effective mass of electrons and the thermal energy. The electron density n is then obtained by integrating the product 
# of the density of states and the Fermi-Dirac distribution over energy. The polylogarithm function can be used to 
# evaluate the Fermi-Dirac integral of order 1/2. The effective electron mass for GaAs is 0.067 times the electron rest 
# mass (m0 = 9.109 x 10^-31 kg). The thermal voltage at room temperature is approximately 0.0259 V. The Planck constant 
# is 6.626 x 10^-34 J·s, and the electron charge is 1.602 x 10^-19 C.





def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    # Constants
    m0 = 9.109e-31  # Electron rest mass in kg
    m_eff = 0.067 * m0  # Effective mass of electron in GaAs
    h = 6.626e-34  # Planck's constant in J·s
    q = 1.602e-19  # Electron charge in C
    kT = 0.0259  # Thermal voltage at room temperature in eV

    # Convert Ef from eV to Joules
    Ef_J = Ef * q

    # Calculate the effective density of states in the conduction band (Nc)
    Nc = 2 * ((2 * np.pi * m_eff * kT * q) / (h**2))**(3/2) / (2 * np.pi**2)

    # Calculate the Fermi-Dirac integral of order 1/2 using the polylog function
    # The polylog function of order 1/2 is used to evaluate the integral
    fermi_integral_half = polylog(1.5, np.exp(Ef_J / (kT * q)))

    # Calculate the electron density n
    n = Nc * fermi_integral_half

    # Convert n from m^-3 to cm^-3
    n_cm3 = n * 1e-6

    return n_cm3


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('36.2', 3)
target = targets[0]

assert np.allclose(fermi_dirac_integral_half_polylog(-0.1), target)
target = targets[1]

assert np.allclose(fermi_dirac_integral_half_polylog(0), target)
target = targets[2]

assert np.allclose(fermi_dirac_integral_half_polylog(0.05), target)
