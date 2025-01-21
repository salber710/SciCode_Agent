import numpy as np

# Background: 
# In semiconductor physics, the Fermi level represents the energy level at which the probability of being occupied by an electron is 50%. For doped semiconductors, the position of the Fermi level changes relative to the intrinsic energy level (E_i) depending on the type and concentration of doping. The built-in potential (or built-in bias) for the p-type (phi_p) and n-type (phi_n) regions can be calculated using the doping concentrations and the intrinsic carrier density.
# The built-in potential for p-type (phi_p) is calculated as:
# phi_p = (kT/q) * ln(N_a / n_i)
# The built-in potential for n-type (phi_n) is calculated as:
# phi_n = (kT/q) * ln(N_d / n_i)
# Here, kT/q is the thermal voltage at room temperature, which is approximately 0.0259V.


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

    # Thermal potential at room temperature
    Vt = 0.0259  # volts

    # Calculate the built-in bias for p-type region (phi_p)
    phi_p = Vt * np.log(N_a / n_i)

    # Calculate the built-in bias for n-type region (phi_n)
    phi_n = Vt * np.log(N_d / n_i)

    return phi_p, phi_n



# Background: 
# In semiconductor devices, the depletion region is the area around the p-n junction where mobile charge carriers are depleted. The depletion widths on the n-type (x_n) and p-type (x_p) sides can be calculated using Poisson's equation. 
# The total depletion width is the sum of x_n and x_p, which can be calculated based on the built-in potential and the doping concentrations. 
# Poisson's equation relates the electric field and charge density, allowing us to compute the depletion widths based on the charge balance.
# The depletion width x_n on the n-type side and x_p on the p-type side are given by:
#   x_n = sqrt((2 * ε * φ_bi * N_a) / (q * (N_a + N_d) * N_d))
#   x_p = sqrt((2 * ε * φ_bi * N_d) / (q * (N_a + N_d) * N_a))
# where φ_bi is the built-in potential (sum of phi_p and phi_n), ε is the permittivity of the material (ε = ε_r * ε_0), 
# ε_0 is the vacuum permittivity, and q is the electron charge.


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
    epsilon_0 = 8.854e-14  # vacuum permittivity in F/cm
    q = 1.602e-19  # electron charge in C
    Vt = 0.0259  # thermal voltage in volts at room temperature

    # Calculate the built-in potential (phi_p + phi_n)
    phi_p = Vt * np.log(N_a / n_i)
    phi_n = Vt * np.log(N_d / n_i)
    phi_bi = phi_p + phi_n  # total built-in potential

    # Calculate the permittivity of the semiconductor material
    epsilon = e_r * epsilon_0

    # Calculate the depletion width in the n-type region (x_n)
    xn = np.sqrt((2 * epsilon * phi_bi * N_a) / (q * (N_a + N_d) * N_d))

    # Calculate the depletion width in the p-type region (x_p)
    xp = np.sqrt((2 * epsilon * phi_bi * N_d) / (q * (N_a + N_d) * N_a))

    return xn, xp

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('34.2', 3)
target = targets[0]

assert np.allclose(depletion(2*10**17,3*10**17,10**12,15), target)
target = targets[1]

assert np.allclose(depletion(1*10**17,2*10**17,10**12,15), target)
target = targets[2]

assert np.allclose(depletion(2*10**17,3*10**17,2*10**11,15), target)
