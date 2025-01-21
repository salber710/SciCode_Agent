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
# In semiconductor physics, the depletion region forms at the junction of n-type and p-type materials. 
# This region is devoid of free charge carriers and acts as an insulator. The width of the depletion 
# region on the n-side (x_n) and p-side (x_p) can be calculated using Poisson's equation, which relates 
# the electric field within the depletion region to the charge density. 
# The total depletion width (W) is given by:
# W = sqrt((2 * epsilon_0 * epsilon_r * V_bi) / (q * (1/N_a + 1/N_d)))
# Where V_bi is the built-in potential (phi_p + phi_n), q is the electron charge (1.6 * 10^-19 C), 
# epsilon_0 is the vacuum permittivity (8.854 * 10^-14 F/cm), and epsilon_r is the relative permittivity.
# The individual depletion widths are given by:
# x_n = (N_a / (N_a + N_d)) * W
# x_p = (N_d / (N_a + N_d)) * W


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
    q = 1.6e-19  # Charge of electron in C
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    Vt = 0.0259  # Thermal voltage at room temperature in V

    # Calculate built-in biases using the previous function's logic
    phi_p = Vt * np.log(N_a / n_i)
    phi_n = Vt * np.log(N_d / n_i)

    # Built-in potential
    V_bi = phi_p + phi_n

    # Total depletion width
    W = np.sqrt((2 * epsilon_0 * e_r * V_bi) / (q * (1/N_a + 1/N_d)))

    # Individual depletion widths
    xn = (N_a / (N_a + N_d)) * W
    xp = (N_d / (N_a + N_d)) * W

    return xn, xp

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('34.2', 3)
target = targets[0]

assert np.allclose(depletion(2*10**17,3*10**17,10**12,15), target)
target = targets[1]

assert np.allclose(depletion(1*10**17,2*10**17,10**12,15), target)
target = targets[2]

assert np.allclose(depletion(2*10**17,3*10**17,2*10**11,15), target)
