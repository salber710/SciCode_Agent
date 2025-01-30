import numpy as np
from scipy.integrate import solve_bvp



# Background: In fiber lasers, the rate equations describe how the intensities of the pump and signal light evolve along the length of the fiber. These equations account for the absorption and emission processes of ions in the fiber, which are characterized by cross-sections (sigma_ap, sigma_ep for the pump and sigma_as, sigma_es for the signal). The overlap factors (gamma_p, gamma_s) represent the efficiency of interaction between the light and the doped ions. The saturation powers (Pssat, Ppsat) indicate the power levels at which the gain starts to saturate. Loss coefficients (alpha_p, alpha_s) account for intrinsic losses in the fiber. The rate of change of the forward and backward pump and signal intensities is determined by these parameters and the spatial variable z.

def f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s):
    '''System of differential equations representing the rate equations of the fiber laser.
    Parameters:
    z : float
        Spatial variable along the fiber's length, representing the position.
    y : ndarray
        Array containing the power values of [forward pump, backward pump, forward signal, backward signal].
    Pssat : float
        Saturation power for the signal.
    Ppsat : float
        Saturation power for the pump.
    N : float
        Total ion population in the fiber.
    sigma_ap : float
        Absorption cross-section for the pump.
    sigma_ep : float
        Emission cross-section for the pump.
    sigma_as : float
        Absorption cross-section for the signal.
    sigma_es : float
        Emission cross-section for the signal.
    gamma_p : float
        Gain coefficient for the pump.
    alpha_p : float
        Loss coefficient for the pump.
    gamma_s : float
        Gain coefficient for the signal.
    alpha_s : float
        Loss coefficient for the signal.
    Returns: ndarray
    The rate of change of the power values for the pump and signal along the fiber:
        dydz[0]: Rate of change of forward pump power.
        dydz[1]: Rate of change of backward pump power.
        dydz[2]: Rate of change of forward signal power.
        dydz[3]: Rate of change of backward signal power.
    '''
    
    # Unpack the intensities from the input array y
    Pf, Pb, Sf, Sb = y
    
    # Calculate the inversion population density
    # N2 is the population of excited ions
    N2 = N * (sigma_ap * (Pf + Pb) + sigma_as * (Sf + Sb)) / (
        (sigma_ap + sigma_ep) * (Pf + Pb) + (sigma_as + sigma_es) * (Sf + Sb) + Ppsat + Pssat
    )
    
    # Calculate the rate of change of the forward pump power
    dPf_dz = -gamma_p * (sigma_ap * (N - N2) * Pf + sigma_ep * N2 * Pf) - alpha_p * Pf
    
    # Calculate the rate of change of the backward pump power
    dPb_dz = gamma_p * (sigma_ap * (N - N2) * Pb + sigma_ep * N2 * Pb) - alpha_p * Pb
    
    # Calculate the rate of change of the forward signal power
    dSf_dz = gamma_s * (sigma_es * N2 * Sf - sigma_as * (N - N2) * Sf) - alpha_s * Sf
    
    # Calculate the rate of change of the backward signal power
    dSb_dz = -gamma_s * (sigma_es * N2 * Sb - sigma_as * (N - N2) * Sb) - alpha_s * Sb
    
    # Return the rate of change as an array
    dydz = np.array([dPf_dz, dPb_dz, dSf_dz, dSb_dz])
    
    return dydz

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('43.1', 3)
target = targets[0]

lambda_s = 1100e-9  # Signal wavelength in meters
lambda_p = 974e-9  # Pump wavelength in meters
tau = 0.8e-3  # Lifetime in seconds
sigma_ap = 26e-21 * 1e-4  # Absorption cross-section for pump in square meters
sigma_ep = 26e-21 * 1e-4  # Emission cross-section for pump in square meters
sigma_as = 1e-23 * 1e-4  # Absorption cross-section for signal in square meters
sigma_es = 1.6e-21 * 1e-4  # Emission cross-section for signal in square meters
A_c = 3.1416e-10  # Core area in square meters
N = 5.5351e25  # Ion concentration in ions/m^3
alpha_p = 2e-3  # Pump loss coefficient in 1/m
alpha_s = 4e-4  # Signal loss coefficient in 1/m
gamma_s = 0.82  # Fraction of the signal mode overlap with the ions
gamma_p = 0.0024  # Fraction of the pump mode overlap with the ions
R1 = 0.99  # Reflectivity of the input mirror
R2 = 0.035  # Reflectivity of the output mirror
L = 40  # Fiber length in meters
Ppl = 50 # Left pump power in watts
Ppr = 50  # Right pump power in watts
c = 3e8  # Speed of light in m/s
h = 6.626e-34  # Planck's constant in J*s
nu_s = c / lambda_s
nu_p = c / lambda_p
# Define functions to calculate saturation powers
def calculate_Pssat(h, nu_s, A_c, gamma_s, sigma_es, sigma_as, tau):
    """Calculate saturation power for the signal.
    Uses the Planck's constant, frequency of the signal, core area, overlap of the signal with the ions,
    emission and absorption cross-sections, and the lifetime of the excited state."""
    Pssat = h * nu_s * A_c / (gamma_s * (sigma_es + sigma_as) * tau)
    return Pssat
def calculate_Ppsat(h, nu_p, A_c, gamma_p, sigma_ep, sigma_ap, tau):
    """Calculate saturation power for the pump."""
    Ppsat = h * nu_p * A_c / (gamma_p * (sigma_ep + sigma_ap) * tau)
    return Ppsat
# Calculate saturation powers
Pssat = calculate_Pssat(h, nu_s, A_c, gamma_s, sigma_es, sigma_as, tau)
Ppsat = calculate_Ppsat(h, nu_p, A_c, gamma_p, sigma_ep, sigma_ap, tau)
z = np.linspace(0, L, 10)
y_guess = np.zeros((4, len(z)))
y_guess[0, :] = Ppl  # Assume initial forward pump power
y_guess[1, :] = Ppr  # Assume initial backward pump power
y_guess[2, :] = 30   # Initial guess for the forward signal power
y_guess[3, :] = Ppr  # Initial guess for the backward signal power
y=y_guess
assert np.allclose(f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s), target)
target = targets[1]

lambda_s = 1100e-9  # Signal wavelength in meters
lambda_p = 974e-9  # Pump wavelength in meters
tau = 0.8e-3  # Lifetime in seconds
sigma_ap = 26e-21 * 1e-4  # Absorption cross-section for pump in square meters
sigma_ep = 26e-21 * 1e-4  # Emission cross-section for pump in square meters
sigma_as = 1e-23 * 1e-4  # Absorption cross-section for signal in square meters
sigma_es = 1.6e-21 * 1e-4  # Emission cross-section for signal in square meters
A_c = 3.1416e-10  # Core area in square meters
N = 5.5351e25  # Ion concentration in ions/m^3
alpha_p = 2e-3  # Pump loss coefficient in 1/m
alpha_s = 4e-4  # Signal loss coefficient in 1/m
gamma_s = 0.82  # Fraction of the signal mode overlap with the ions
gamma_p = 0.0024  # Fraction of the pump mode overlap with the ions
R1 = 0.99  # Reflectivity of the input mirror
R2 = 0.035  # Reflectivity of the output mirror
L = 40  # Fiber length in meters
Ppl = 50 # Left pump power in watts
Ppr = 50  # Right pump power in watts
c = 3e8  # Speed of light in m/s
h = 6.626e-34  # Planck's constant in J*s
nu_s = c / lambda_s
nu_p = c / lambda_p
# Define functions to calculate saturation powers
def calculate_Pssat(h, nu_s, A_c, gamma_s, sigma_es, sigma_as, tau):
    """Calculate saturation power for the signal.
    Uses the Planck's constant, frequency of the signal, core area, overlap of the signal with the ions,
    emission and absorption cross-sections, and the lifetime of the excited state."""
    Pssat = h * nu_s * A_c / (gamma_s * (sigma_es + sigma_as) * tau)
    return Pssat
def calculate_Ppsat(h, nu_p, A_c, gamma_p, sigma_ep, sigma_ap, tau):
    """Calculate saturation power for the pump."""
    Ppsat = h * nu_p * A_c / (gamma_p * (sigma_ep + sigma_ap) * tau)
    return Ppsat
# Calculate saturation powers
Pssat = calculate_Pssat(h, nu_s, A_c, gamma_s, sigma_es, sigma_as, tau)
Ppsat = calculate_Ppsat(h, nu_p, A_c, gamma_p, sigma_ep, sigma_ap, tau)
z = np.linspace(0, L, 10)
y_guess = np.zeros((4, len(z)))
y_guess[0, :] = Ppl  # Assume initial forward pump power
y_guess[1, :] = Ppr  # Assume initial backward pump power
y_guess[2, :] = 50   # Initial guess for the forward signal power
y_guess[3, :] = Ppr  # Initial guess for the backward signal power
y=y_guess
assert np.allclose(f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s), target)
target = targets[2]

lambda_s = 1100e-9  # Signal wavelength in meters
lambda_p = 974e-9  # Pump wavelength in meters
tau = 0.8e-3  # Lifetime in seconds
sigma_ap = 26e-21 * 1e-4  # Absorption cross-section for pump in square meters
sigma_ep = 26e-21 * 1e-4  # Emission cross-section for pump in square meters
sigma_as = 1e-23 * 1e-4  # Absorption cross-section for signal in square meters
sigma_es = 1.6e-21 * 1e-4  # Emission cross-section for signal in square meters
A_c = 3.1416e-10  # Core area in square meters
N = 5.5351e25  # Ion concentration in ions/m^3
alpha_p = 2e-3  # Pump loss coefficient in 1/m
alpha_s = 4e-4  # Signal loss coefficient in 1/m
gamma_s = 0.82  # Fraction of the signal mode overlap with the ions
gamma_p = 0.0024  # Fraction of the pump mode overlap with the ions
R1 = 0.99  # Reflectivity of the input mirror
R2 = 0.035  # Reflectivity of the output mirror
L = 40  # Fiber length in meters
Ppl = 50 # Left pump power in watts
Ppr = 50  # Right pump power in watts
c = 3e8  # Speed of light in m/s
h = 6.626e-34  # Planck's constant in J*s
nu_s = c / lambda_s
nu_p = c / lambda_p
# Define functions to calculate saturation powers
def calculate_Pssat(h, nu_s, A_c, gamma_s, sigma_es, sigma_as, tau):
    """Calculate saturation power for the signal.
    Uses the Planck's constant, frequency of the signal, core area, overlap of the signal with the ions,
    emission and absorption cross-sections, and the lifetime of the excited state."""
    Pssat = h * nu_s * A_c / (gamma_s * (sigma_es + sigma_as) * tau)
    return Pssat
def calculate_Ppsat(h, nu_p, A_c, gamma_p, sigma_ep, sigma_ap, tau):
    """Calculate saturation power for the pump."""
    Ppsat = h * nu_p * A_c / (gamma_p * (sigma_ep + sigma_ap) * tau)
    return Ppsat
# Calculate saturation powers
Pssat = calculate_Pssat(h, nu_s, A_c, gamma_s, sigma_es, sigma_as, tau)
Ppsat = calculate_Ppsat(h, nu_p, A_c, gamma_p, sigma_ep, sigma_ap, tau)
z = np.linspace(0, L, 10)
y_guess = np.zeros((4, len(z)))
y_guess[0, :] = Ppl  # Assume initial forward pump power
y_guess[1, :] = Ppr  # Assume initial backward pump power
y_guess[2, :] = 70   # Initial guess for the forward signal power
y_guess[3, :] = Ppr  # Initial guess for the backward signal power
y=y_guess
assert np.allclose(f(z, y, Pssat, Ppsat, N, sigma_ap, sigma_ep, sigma_as, sigma_es, gamma_p, alpha_p, gamma_s, alpha_s), target)
