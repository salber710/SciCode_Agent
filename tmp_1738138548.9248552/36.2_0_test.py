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



# Background: 
# The Fermi-Dirac distribution describes the occupancy of electron energy states in a semiconductor at thermal equilibrium. 
# The electron density (n) in the conduction band can be calculated by integrating the product of the density of states (DOS) 
# and the Fermi-Dirac distribution over energy. For a semiconductor like GaAs, the effective density of states in the conduction 
# band is influenced by the effective mass of electrons. The Fermi-Dirac integral of order 1/2 is used to compute the electron 
# density, which can be expressed using the polylogarithm function. The effective mass of electrons in GaAs is 0.067 times the 
# free electron mass (m0 = 9.109 x 10^-31 kg). The thermal voltage at room temperature (approximately 300K) is 0.0259 V. 
# The electron charge is 1.602 x 10^-19 C. The Planck constant is 6.626 x 10^-34 J·s. The integral is performed over energy 
# levels from the conduction band edge to infinity, and the result is converted to electron density in cm^-3.




def fermi_dirac_integral_half_polylog(Ef):
    '''Function to compute the Fermi-Dirac integral of order 1/2 using polylog
    Input:
    Ef (float): Fermi level compared to the conduction band (eV)
    Output:
    n (float): electron density (cm^-3)
    '''
    
    # Constants
    m0 = 9.109e-31  # Free electron mass in kg
    m_eff = 0.067 * m0  # Effective mass of electron in GaAs
    h = 6.626e-34  # Planck constant in J·s
    q = 1.602e-19  # Electron charge in C
    kT = 0.0259  # Thermal voltage at room temperature in eV
    kT_J = kT * q  # Convert thermal voltage to Joules

    # Effective density of states in the conduction band (Nc)
    Nc = 2 * ((2 * np.pi * m_eff * kT_J) / (h**2))**(3/2) / (2 * np.pi**2)

    # Fermi-Dirac integral of order 1/2
    def fermi_dirac_integral_half(E):
        return np.sqrt(E) / (1 + np.exp(E - Ef))

    # Integrate from 0 to infinity
    integral_result, _ = quad(fermi_dirac_integral_half, 0, np.inf)

    # Calculate electron density
    n = Nc * integral_result

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