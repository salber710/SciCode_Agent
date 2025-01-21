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


# Background: In a semiconductor junction, the depletion region is formed due to the redistribution of charge carriers.
# This region becomes void of mobile charge carriers, leading to the establishment of an electric field. The width of
# the depletion region can be calculated using Poisson's equation, which relates the built-in potential to the doping
# concentrations and permittivity. For a pn-junction, the depletion widths on the n-type side (x_n) and p-type side (x_p)
# can be determined by:
#   x_n = sqrt((2 * epsilon_0 * epsilon_r * phi_bi) / (q * (N_a + N_d) * N_d))
#   x_p = sqrt((2 * epsilon_0 * epsilon_r * phi_bi) / (q * (N_a + N_d) * N_a))
# where phi_bi is the built-in potential difference across the junction, epsilon_0 is the vacuum permittivity, 
# epsilon_r is the relative permittivity of the material, and q is the electron charge.


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

    # Thermal potential at room temperature
    Vt = 0.0259  # volts

    # Calculate the built-in potential (phi_bi)
    phi_p = Vt * np.log(N_a / n_i)
    phi_n = Vt * np.log(N_d / n_i)
    phi_bi = phi_p + phi_n  # Total built-in potential

    # Calculate depletion widths using Poisson's equation
    xn = np.sqrt((2 * epsilon_0 * e_r * phi_bi) / (q * (N_a + N_d) * N_d))
    xp = np.sqrt((2 * epsilon_0 * e_r * phi_bi) / (q * (N_a + N_d) * N_a))

    return xn, xp



# Background: In a pn-junction, the potential distribution across the depletion region is crucial for understanding
# the behavior of the junction. The conduction band potential starts from 0V at the p-type side and increases
# across the depletion region. The potential change can be represented as a linear gradient due to the
# electric field across the junction, which is derived from the built-in potential. To compute the potential
# diagram, we first need to determine the depletion widths (x_n and x_p) using the doping concentrations (N_a and N_d),
# intrinsic carrier density (n_i), and relative permittivity (e_r). The potential distribution is then calculated
# in increments of 0.1 nm from the start of the depletion region at the p-type side.


def potential(N_a, N_d, n_i, e_r):
    '''Inputs:
    N_a: float, doping concentration in n-type region # cm^{-3}
    N_d: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    potential: narray, the potential distribution
    '''

    # Constants
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19  # Electron charge in C
    Vt = 0.0259  # Thermal potential at room temperature in volts
    
    # Calculate the built-in potential (phi_bi)
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    phi_bi = phi_p + phi_n  # Total built-in potential

    # Calculate depletion widths using the previous function
    xn, xp = depletion(N_a, N_d, n_i, e_r)

    # Calculate the total depletion width
    W = xn + xp

    # Define the space increment (0.1 nm = 0.00001 cm)
    dx = 0.00001  # cm

    # Create an array of positions from 0 to W with increment dx
    x_values = np.arange(0, W, dx)

    # Calculate the linear potential distribution across the depletion region
    # The potential starts at 0V at the beginning of the p-type region (x = 0)
    potential = (phi_bi / W) * x_values

    return xn, xp, potential

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('34.3', 4)
target = targets[0]

xn,xp,_ = potential(2*10**17,2*10**17,10**11,15)
assert (xn==xp) == target
target = targets[1]

assert np.allclose(potential(1*10**18,2*10**18,10**11,15)[2], target)
target = targets[2]

assert np.allclose(potential(1*10**17,2*10**17,10**11,15)[2], target)
target = targets[3]

assert np.allclose(potential(1*10**18,2*10**17,10**11,10)[2], target)
