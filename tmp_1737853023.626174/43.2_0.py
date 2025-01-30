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
    
    # Ensure non-negative intensities for physical correctness
    Pf, Pb, Sf, Sb = np.abs([Pf, Pb, Sf, Sb])
    
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



# Background: In fiber laser systems, boundary conditions are crucial for defining how the laser operates at the start and end of the fiber. These conditions ensure that the power levels of the pump and signal light are consistent with the physical setup of the laser. The input and output mirrors of the laser have specific reflectivities (R1 and R2), which determine how much light is reflected back into the fiber. The boundary conditions must account for the input power of the pumps (Ppl and Ppr) and the reflectivity of the mirrors to ensure the laser operates correctly. The conditions are defined such that the power at the start and end of the fiber matches the expected values based on these parameters.

def bc(ya, yb, Ppl, Ppr, R1, R2):
    '''Define the boundary conditions for the fiber laser.
    Parameters:
    ya : ndarray
        Array of power values at the start of the fiber. Contains values corresponding to:
        ya[0] - Power of the forward pump at the fiber input.
        ya[1] - Power of the backward pump at the fiber input.
        ya[2] - Power of the forward signal at the fiber input.
        ya[3] - Power of the backward signal at the fiber input.
    yb : ndarray
        Array of power values at the end of the fiber. Contains values corresponding to:
        yb[0] - Power of the forward pump at the fiber output.
        yb[1] - Power of the backward pump at the fiber output.
        yb[2] - Power of the forward signal at the fiber output.
        yb[3] - Power of the backward signal at the fiber output.
    Ppl : float
        Input power for the left pump, affecting the starting boundary of the laser.
    Ppr : float
        Input power for the right pump, affecting the ending boundary of the laser.
    R1 : float
        Reflectivity of the input mirror, modifying the behavior of the light at the fiber's start.
    R2 : float
        Reflectivity of the output mirror, modifying the behavior of the light at the fiber's end.
    Returns:
    ndarray
        An array of four boundary conditions calculated as follows:
        bc[0]: boundary condition for rate of change of forward pump power.
        bc[1]: boundary condition for rate of change of backward pump power.
        bc[2]: boundary condition for rate of change of forward signal power.
        bc[3]: boundary condition for rate of change of backward signal power.
    '''
    
    # Calculate the boundary conditions
    bc = np.array([
        ya[0] - Ppl,          # Difference between initial forward pump power and left pump power
        yb[1] - Ppr,          # Difference between final backward pump power and right pump power
        ya[2] - R1 * ya[3],   # Reflectivity condition at the input mirror for the signal
        yb[3] - R2 * yb[2]    # Reflectivity condition at the output mirror for the signal
    ])
    
    return bc

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('43.2', 3)
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
ya = [0,0,0,0]
yb= [0, 0, 0, 0]
assert np.allclose(bc(ya, yb, Ppl, Ppr, R1, R2), target)
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
ya = [1, 0.5, 2, 1]
yb = [0.5, 1, 1, 2]
assert np.allclose(bc(ya, yb, Ppl, Ppr, R1, R2), target)
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
ya = [1, 1, 2, 1]
yb = [0.5, 1, 2, 2]
assert np.allclose(bc(ya, yb, Ppl, Ppr, R1, R2), target)
