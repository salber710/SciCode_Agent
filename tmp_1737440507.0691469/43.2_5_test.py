import numpy as np
from scipy.integrate import solve_bvp

# Background: 
# In fiber lasers, the propagation of pump and signal light within the fiber is governed by rate equations.
# These equations model the change in light intensities as they move through the fiber. Critical to these
# equations are the absorption and emission cross-sections, which represent the probabilities of absorption
# and emission of photons by the ions in the fiber. The overlap factors (gamma_p and gamma_s) quantify how well
# the pump and signal fields overlap with the doped ion distribution. Saturation powers (Ppsat and Pssat) define
# the power levels at which the gain begins to saturate, limiting further amplification. The loss coefficients 
# (alpha_p and alpha_s) account for intrinsic losses in the fiber, such as scattering. By solving these rate
# equations, we can predict the spatial evolution of the forward and backward pump and signal powers along the fiber.


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

    # Unpack the current values of forward/backward pumps and signals
    Pf_forward, Pf_backward, Ps_forward, Ps_backward = y

    # Calculate the inversion of the population
    # N2 is the upper state population density, N is the total population density
    # Gain and absorption depend on the population inversion
    N2 = N * (Ps_forward / Pssat + Ps_backward / Pssat + Pf_forward / Ppsat + Pf_backward / Ppsat) / (
            1 + Ps_forward / Pssat + Ps_backward / Pssat + Pf_forward / Ppsat + Pf_backward / Ppsat)

    # Define the rate equations based on gain and loss
    dPf_forward_dz = -gamma_p * (sigma_ap * (N - N2) - sigma_ep * N2) * Pf_forward - alpha_p * Pf_forward
    dPf_backward_dz = gamma_p * (sigma_ap * (N - N2) - sigma_ep * N2) * Pf_backward - alpha_p * Pf_backward

    dPs_forward_dz = gamma_s * (sigma_es * N2 - sigma_as * (N - N2)) * Ps_forward - alpha_s * Ps_forward
    dPs_backward_dz = -gamma_s * (sigma_es * N2 - sigma_as * (N - N2)) * Ps_backward - alpha_s * Ps_backward

    # Return the derivatives as an array
    dydz = np.array([dPf_forward_dz, dPf_backward_dz, dPs_forward_dz, dPs_backward_dz])

    return dydz



# Background: In fiber laser systems, boundary conditions are critical for ensuring the laser operates correctly
# along its length. These conditions define how power values at the start and end of the fiber relate to the input
# pump powers and mirror reflectivities. Specifically, the boundary conditions ensure that the forward and backward
# pump powers match the input powers at the fiber's boundaries, and the reflectivities of the mirrors are considered
# in the signal power reflections. The reflectivity of the input mirror affects the backward signal at the start,
# while the output mirror affects the forward signal at the end. These conditions are fundamental for setting up
# the boundary value problem (BVP) that models the laser's operation.

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
    bc = np.array([
        ya[0] - Ppl,            # Forward pump power at start equals left pump power
        yb[1] - Ppr,            # Backward pump power at end equals right pump power
        ya[3] - R1 * ya[2],     # Backward signal power at start equals reflectivity times forward signal
        yb[2] - R2 * yb[3]      # Forward signal power at end equals reflectivity times backward signal
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
