import numpy as np

# Background: In semiconductor physics, the built-in potential (or built-in bias) is the potential difference across a p-n junction in thermal equilibrium. 
# It arises due to the difference in the Fermi levels of the n-type and p-type regions. The built-in potential can be calculated using the formula:
# phi_p = V_t * ln(N_a / n_i) for the p-type region and phi_n = V_t * ln(N_d / n_i) for the n-type region, where V_t is the thermal voltage.
# At room temperature (approximately 300K), the thermal voltage V_t is approximately 0.0259 V.


def Fermi(N_a, N_d, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    V_t = 0.0259  # Thermal potential at room temperature in volts

    if N_a <= 0 or N_d <= 0 or n_i <= 0:
        raise ValueError("Doping concentrations and intrinsic carrier density must be positive.")

    # Calculate the built-in bias for the p-type region
    phi_p = V_t * np.log(N_a / n_i)

    # Calculate the built-in bias for the n-type region
    phi_n = V_t * np.log(N_d / n_i)

    return phi_p, phi_n



# Background: In semiconductor physics, the depletion region is the area around the p-n junction where mobile charge carriers are depleted. 
# The width of the depletion region can be calculated using Poisson's equation, which relates the electric field to the charge density. 
# The depletion width on the n-type side (x_n) and the p-type side (x_p) can be determined using the built-in potential and the doping concentrations.
# The formulas for the depletion widths are:
# x_n = sqrt((2 * epsilon * phi_p) / (q * (N_a + N_d) * N_d))
# x_p = sqrt((2 * epsilon * phi_n) / (q * (N_a + N_d) * N_a))
# where epsilon is the permittivity of the material (epsilon = epsilon_r * epsilon_0), q is the electron charge, and phi_p and phi_n are the built-in potentials.


def depletion(N_a, N_d, n_i, e_r):
    '''This function calculates the depletion width in both n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    '''
    # Constants
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19  # Electron charge in C
    V_t = 0.0259  # Thermal potential at room temperature in volts

    # Calculate the built-in potentials
    phi_p = V_t * np.log(N_a / n_i)
    phi_n = V_t * np.log(N_d / n_i)

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0

    # Calculate the depletion widths
    xn = np.sqrt((2 * epsilon * phi_p) / (q * (N_a + N_d) * N_d))
    xp = np.sqrt((2 * epsilon * phi_n) / (q * (N_a + N_d) * N_a))

    return xn, xp

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('34.2', 3)
target = targets[0]

assert np.allclose(depletion(2*10**17,3*10**17,10**12,15), target)
target = targets[1]

assert np.allclose(depletion(1*10**17,2*10**17,10**12,15), target)
target = targets[2]

assert np.allclose(depletion(2*10**17,3*10**17,2*10**11,15), target)
