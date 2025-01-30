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
    c = 3e8  # Speed of light in m/s
    h = 6.626e-34  # Planck's constant in J·s
    q = 1.602e-19  # Charge of an electron in Coulombs
    
    # Convert lambda_i from nm to meters
    lambda_i_m = lambda_i * 1e-9
    
    # Convert A from μm^2 to m^2
    A_m2 = A * 1e-12
    
    # Calculate photon energy (E_photon) in Joules
    E_photon = h * c / lambda_i_m
    
    # Calculate photon flux (phi) in photons per second per m^2
    phi = P / (E_photon * A_m2)
    
    # Convert alpha from cm^-1 to m^-1
    alpha_m = alpha * 100
    
    # Convert x from μm to m
    x_m = x * 1e-6
    
    # Calculate the generation rate (G) in carriers per second per m^3
    G = phi * alpha_m * np.exp(-alpha_m * x_m)
    
    # Calculate the generated carrier density (dN) in cm^-3
    # Convert G from m^-3 to cm^-3 by multiplying by 1e-6
    dN = G * tau * 1e-6
    
    return dN



def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    
    # Constants
    kT = 0.0259  # Thermal voltage at room temperature in eV
    m_e = 0.067 * 9.109e-31  # Effective electron mass for GaAs in kg
    h_bar = 1.0545718e-34  # Reduced Planck constant in J·s
    q = 1.602e-19  # Electron charge in Coulombs

    # Conversion factor from m^-3 to cm^-3
    conversion_factor = 1e-6

    # Calculation of the prefactor for the density of states integral
    prefactor = (2 * (2 * np.pi * m_e * kT * q) ** 1.5) / (h_bar ** 3 * np.pi ** 2)

    # Define the integrand for the Fermi-Dirac integral of order 1/2
    def integrand(E):
        return np.sqrt(E) / (1 + np.exp(E - Ef))

    # Perform the integral from 0 to infinity
    integral, _ = quad(integrand, 0, np.inf)

    # Calculate the electron density
    n = prefactor * integral * conversion_factor

    return n


try:
    targets = process_hdf5_to_tuple('36.2', 3)
    target = targets[0]
    assert np.allclose(fermi_dirac_integral_half_polylog(-0.1), target)

    target = targets[1]
    assert np.allclose(fermi_dirac_integral_half_polylog(0), target)

    target = targets[2]
    assert np.allclose(fermi_dirac_integral_half_polylog(0.05), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e