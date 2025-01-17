import numpy as np
from scipy.integrate import solve_bvp

# Background: 
# In fiber lasers, the rate equations describe how the intensities of the pump and signal light evolve along the length of the fiber. 
# These equations account for the absorption and emission processes that occur as light interacts with the doped ions in the fiber. 
# The pump light is absorbed by the ions, which can then emit signal light through stimulated emission. 
# The rate of change of the pump and signal intensities is influenced by several factors:
# - Saturation powers (Pssat, Ppsat) determine the intensity at which the gain or absorption saturates.
# - Cross-sections (sigma_ap, sigma_ep, sigma_as, sigma_es) represent the probability of absorption and emission events for the pump and signal.
# - Overlap factors (gamma_p, gamma_s) describe how well the pump and signal modes overlap with the doped region of the fiber.
# - Loss coefficients (alpha_p, alpha_s) account for intrinsic losses in the fiber.
# The function f calculates the spatial derivatives of the forward and backward pump and signal intensities, which are used to model the laser's behavior.

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

    # Calculate the population inversion density
    Wp = (Pf + Pb) / Ppsat
    Ws = (Sf + Sb) / Pssat
    N2 = N * (Wp * sigma_ap + Ws * sigma_as) / ((Wp * sigma_ap + Ws * sigma_as) + (1 + Wp) * sigma_ep + (1 + Ws) * sigma_es)
    N1 = N - N2

    # Calculate the rate of change of the forward and backward pump powers
    dPf_dz = -gamma_p * (sigma_ap * N1 - sigma_ep * N2) * Pf - alpha_p * Pf
    dPb_dz = gamma_p * (sigma_ap * N1 - sigma_ep * N2) * Pb - alpha_p * Pb

    # Calculate the rate of change of the forward and backward signal powers
    dSf_dz = gamma_s * (sigma_es * N2 - sigma_as * N1) * Sf - alpha_s * Sf
    dSb_dz = -gamma_s * (sigma_es * N2 - sigma_as * N1) * Sb - alpha_s * Sb

    # Return the derivatives as an array
    dydz = np.array([dPf_dz, dPb_dz, dSf_dz, dSb_dz])

    return dydz



# Background: In fiber laser systems, boundary conditions are crucial for defining how the laser operates at the start and end of the fiber. 
# These conditions ensure that the power levels of the pump and signal light are consistent with the physical setup of the laser. 
# The boundary conditions involve the input and output power levels of the pumps and signals, as well as the reflectivities of the mirrors at both ends of the fiber. 
# The reflectivity of the mirrors affects how much light is reflected back into the fiber, influencing the laser's efficiency and stability. 
# The function `bc` is designed to establish these boundary conditions by comparing the power levels at the start and end of the fiber with the input pump powers and the reflectivities of the mirrors.

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
        bc[0]: Difference between the initial forward pump power and the left pump power.
        bc[1]: Difference between the final backward pump power and the right pump power.
        bc[2]: Difference between the initial forward signal power and the product of the reflectivity of the input mirror and the initial backward signal power.
        bc[3]: Difference between the final backward signal power and the product of the reflectivity of the output mirror and the final forward signal power.
    '''

    # Calculate the boundary conditions
    bc = np.array([
        ya[0] - Ppl,                # Forward pump power at start minus left pump power
        yb[1] - Ppr,                # Backward pump power at end minus right pump power
        ya[2] - R1 * ya[3],         # Forward signal power at start minus R1 times backward signal power at start
        yb[3] - R2 * yb[2]          # Backward signal power at end minus R2 times forward signal power at end
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
