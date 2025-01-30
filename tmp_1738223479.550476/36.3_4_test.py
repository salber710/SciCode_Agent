from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton

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
    h = 6.626e-34  # Planck constant in J·s
    c = 3e8        # Speed of light in m/s

    # Convert wavelength from nm to m and area from μm^2 to cm^2
    lambda_m = lambda_i * 1e-9
    A_cm2 = A * 1e-8

    # Calculate photon energy in electronvolts (eV) instead of Joules
    E_photon_eV = (h * c / lambda_m) / 1.60218e-19  # Convert J to eV

    # Convert depth from μm to cm
    x_cm = x * 1e-4

    # Calculate photon flux density (photons per cm^2 per s) using an alternative method
    n_photons = P / (E_photon_eV * 1.60218e-19)  # Photon energy in eV converted back to J

    # Use a different approach to calculate photon flux density
    photon_flux_density = n_photons / (A_cm2 * (1 + alpha * x_cm))

    # Compute generation rate using a quadratic attenuation model
    G = alpha * photon_flux_density * (1 - (alpha * x_cm)**2 / 2)

    # Calculate excess carrier density considering the carrier lifetime
    dN = G * tau

    return dN




def fermi_dirac_integral_half_adaptive_quadrature(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using adaptive Gauss-Kronrod quadrature
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    # Constants
    m0 = m_e  # Electron rest mass in kg
    meff = 0.067 * m0  # Effective mass of electron in GaAs
    T = 300  # Temperature in Kelvin
    Vth = 0.0259  # Thermal voltage at room temperature in V

    # Effective density of states at the conduction band edge (Nc) in cm^-3
    Nc = 2 * ((2 * np.pi * meff * k * T) / (h**2))**(3/2) * (1e-6)  # Convert m^-3 to cm^-3

    # Convert Ef from eV to thermal voltage units
    eta = Ef / Vth

    # Define the integrand for the Fermi-Dirac integral of order 1/2
    def integrand(E):
        return E**0.5 / (1 + np.exp(E - eta))

    # Adaptive Gauss-Kronrod quadrature using numpy's built-in functionality

    F_half, _ = quad(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-9)

    # Calculate electron density
    n = Nc * F_half

    return n






def generation(P, A, lambda_i, alpha, tau, x):
    h = 6.626e-34  # Planck constant in J·s
    c = 3e8        # Speed of light in m/s

    # Convert wavelength from nm to m and area from μm^2 to cm^2
    lambda_m = lambda_i * 1e-9
    A_cm2 = A * 1e-8

    # Calculate photon energy in Joules
    E_photon = h * c / lambda_m

    # Convert depth from μm to cm
    x_cm = x * 1e-4

    # Calculate photon flux density (photons per cm^2 per s)
    photon_flux_density = P / (E_photon * A_cm2)

    # Compute generation rate
    G = alpha * photon_flux_density * np.exp(-alpha * x_cm)

    # Calculate excess carrier density considering the carrier lifetime
    dN = G * tau

    return dN

def inverse_fermi_dirac_integral_half_polylog_brute_force(P, A, lambda_i, alpha, tau, x, n=None):
    m0 = 9.109e-31  # Electron rest mass in kg
    meff = 0.067 * m0  # Effective mass of electron in GaAs
    k = 1.380649e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in Kelvin
    h = 6.626e-34  # Planck constant in J·s
    Vth = 0.0259  # Thermal voltage at room temperature in V
    q = 1.60218e-19  # Electron charge in C

    # Effective density of states at the conduction band edge (Nc) in cm^-3
    Nc = 2 * ((2 * np.pi * meff * k * T) / (h**2))**(3/2) * (1e-6)  # Convert m^-3 to cm^-3

    # If electron density is not provided, calculate it using the generation function
    if n is None:
        n = generation(P, A, lambda_i, alpha, tau, x)

    # Define the function whose root we need to find
    def f(Ef):
        eta = Ef / Vth
        # Define the integrand for the Fermi-Dirac integral of order 1/2
        def integrand(E):
            return E**0.5 / (1 + np.exp(E - eta))
        
        # Compute the Fermi-Dirac integral
        F_half, _ = quad(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-9)
        
        # Compute the electron density for the given Fermi level
        n_calculated = Nc * F_half
        
        # The root of this function is where the calculated electron density matches the given density
        return n_calculated - n

    # Use brute force optimization to minimize the absolute error
    result = minimize_scalar(lambda Ef: abs(f(Ef)), bounds=(-2.0, 2.0), method='bounded')

    return result.x


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