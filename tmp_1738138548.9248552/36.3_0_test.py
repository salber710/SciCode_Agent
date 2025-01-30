from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton

def generation(P, A, lambda_i, alpha, tau, x):


    # Constants
    c = 3e8  # speed of light in m/s
    h = 6.626e-34  # Planck constant in J·s

    # Convert units
    lambda_cm = lambda_i * 1e-7  # Convert wavelength from nm to cm
    A_cm2 = A * 1e-8  # Convert area from μm^2 to cm^2
    x_cm = x * 1e-4  # Convert depth from μm to cm

    # Calculate photon energy in ergs
    E_photon = h * c / lambda_cm * 1e7  # Energy of a single photon in ergs

    # Calculate photon flux (photons per second per cm^2)
    photon_flux = P / (E_photon * A_cm2)

    # Calculate generation rate at depth x using Beer-Lambert law
    G_x = photon_flux * alpha * np.exp(-alpha * x_cm)  # Generation rate in cm^-3·s^-1

    # Calculate generated carrier density
    dN = G_x * tau  # Carrier density in cm^-3

    return dN




def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using a direct computation approach with adaptive quadrature
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    
    # Constants
    m0 = m_e  # Free electron mass in kg
    m_eff = 0.067 * m0  # Effective mass of electron in GaAs
    kT = 0.0259  # Thermal voltage at room temperature in eV
    kT_J = kT * e  # Convert thermal voltage to Joules
    h_bar = h / (2 * pi)  # Reduced Planck constant

    # Effective density of states in the conduction band (Nc)
    Nc = 2 * ((2 * pi * m_eff * kT_J) / (h_bar**2))**(3/2) / (2 * pi**2)

    # Fermi-Dirac integral of order 1/2 using adaptive quadrature

    def integrand(E):
        return (E**0.5) / (np.exp((E - Ef) / kT) + 1)
    integral_result, _ = quad(integrand, 0, np.inf)

    # Calculate electron density
    n = Nc * integral_result * 1e-6  # Convert m^-3 to cm^-3

    return n



# Background: 
# The Fermi-Dirac integral of order 1/2 is used to relate the electron density to the Fermi level in semiconductors.
# The inverse problem involves finding the Fermi level given the electron density. This can be solved using numerical methods
# such as the Newton-Raphson method. The effective density of states (Nc) in the conduction band is crucial for this calculation.
# The generation function calculates the electron density from optical parameters if not provided. The Newton-Raphson method
# iteratively finds the root of the function by approximating the derivative and updating the guess for the Fermi level.




def generation(P, A, lambda_i, alpha, tau, x):
    # Constants
    c = 3e8  # speed of light in m/s
    h = 6.626e-34  # Planck constant in J·s

    # Convert units
    lambda_cm = lambda_i * 1e-7  # Convert wavelength from nm to cm
    A_cm2 = A * 1e-8  # Convert area from μm^2 to cm^2
    x_cm = x * 1e-4  # Convert depth from μm to cm

    # Calculate photon energy in ergs
    E_photon = h * c / lambda_cm * 1e7  # Energy of a single photon in ergs

    # Calculate photon flux (photons per second per cm^2)
    photon_flux = P / (E_photon * A_cm2)

    # Calculate generation rate at depth x using Beer-Lambert law
    G_x = photon_flux * alpha * np.exp(-alpha * x_cm)  # Generation rate in cm^-3·s^-1

    # Calculate generated carrier density
    dN = G_x * tau  # Carrier density in cm^-3

    return dN

def inverse_fermi_dirac_integral_half_polylog_newton(P, A, lambda_i, alpha, tau, x, n=None):
    '''This function uses the Newton-Raphson method to find the root of an implicit function.
    Inputs:
    P (float): incident optical power in W
    A (float): beam area in μm^2
    lambda_i (float): incident wavelength in nm
    alpha (float): absorption coefficient in cm^-1
    tau (float): lifetime of excess carriers in s
    x (float): depth variable in μm
    n (float): electron density, which is unknown at default (set as None)
    Outputs:
    Ef: Fermi level
    '''
    
    # Constants
    m0 = 9.109e-31  # Free electron mass in kg
    m_eff = 0.067 * m0  # Effective mass of electron in GaAs
    kT = 0.0259  # Thermal voltage at room temperature in eV
    e = 1.602e-19  # Electron charge in C
    kT_J = kT * e  # Convert thermal voltage to Joules
    h_bar = 6.626e-34 / (2 * np.pi)  # Reduced Planck constant

    # Effective density of states in the conduction band (Nc)
    Nc = 2 * ((2 * np.pi * m_eff * kT_J) / (h_bar**2))**(3/2) / (2 * np.pi**2)

    # If electron density n is not provided, calculate it using the generation function
    if n is None:
        n = generation(P, A, lambda_i, alpha, tau, x)

    # Define the function for which we want to find the root
    def f(Ef):
        def integrand(E):
            return (E**0.5) / (np.exp((E - Ef) / kT) + 1)
        integral_result, _ = quad(integrand, 0, np.inf)
        return Nc * integral_result * 1e-6 - n  # Convert m^-3 to cm^-3

    # Use the Newton-Raphson method to find the Fermi level Ef
    Ef_initial_guess = 0.1  # Initial guess for the Fermi level in eV
    Ef = newton(f, Ef_initial_guess)

    return Ef


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e