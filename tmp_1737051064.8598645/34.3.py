import numpy as np

# Background: 
# In semiconductor physics, the built-in potential (or built-in bias) is the potential difference across a p-n junction in thermal equilibrium. 
# It arises due to the difference in the Fermi levels of the n-type and p-type regions. 
# The built-in potential can be calculated using the doping concentrations of the n-type (N_d) and p-type (N_a) regions, 
# as well as the intrinsic carrier density (n_i). 
# The formula for the built-in potential (V_bi) is given by:
# V_bi = V_t * ln((N_a * N_d) / n_i^2)
# where V_t is the thermal voltage, which is approximately 0.0259 V at room temperature (300 K).
# The built-in bias for the p-type region (phi_p) and the n-type region (phi_n) can be calculated as:
# phi_p = V_bi / 2
# phi_n = -V_bi / 2
# These represent the energy level differences compared to the intrinsic energy level (E_i).


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
    V_t = 0.0259  # Thermal voltage at room temperature in volts
    V_bi = V_t * np.log((N_a * N_d) / n_i**2)
    phi_p = V_bi / 2
    phi_n = -V_bi / 2

    return phi_p, phi_n


# Background: 
# In semiconductor physics, the depletion region is the area around the p-n junction where mobile charge carriers are depleted.
# The width of the depletion region can be calculated using Poisson's equation, which relates the electric field to the charge density.
# The depletion width on the n-type side (x_n) and the p-type side (x_p) can be determined using the built-in potential (V_bi) and the doping concentrations (N_a and N_d).
# The formulas for the depletion widths are derived from the charge neutrality condition and Poisson's equation:
# x_n = sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_a / N_d))
# x_p = sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_d / N_a))
# where epsilon is the permittivity of the semiconductor material, q is the electron charge, and V_bi is the built-in potential.
# The permittivity epsilon is given by epsilon = epsilon_r * epsilon_0, where epsilon_r is the relative permittivity and epsilon_0 is the vacuum permittivity.


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
    V_t = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the built-in potential V_bi
    V_bi = V_t * np.log((N_a * N_d) / n_i**2)

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0

    # Calculate the depletion widths
    xn = np.sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_a / N_d))
    xp = np.sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_d / N_a))

    return xn, xp



# Background: 
# In semiconductor physics, the potential distribution across a p-n junction can be visualized using a band diagram.
# The conduction band potential is typically set to 0V at the start of the depletion region on the p-type side.
# The potential changes across the depletion region due to the built-in electric field.
# The depletion widths (x_n and x_p) determine the extent of the depletion region on the n-type and p-type sides, respectively.
# To visualize the potential distribution, we can create an array representing the conduction band potential across the depletion region.
# The potential is constant outside the depletion region and changes linearly within it.
# We will use a space increment of 0.1 nm (0.00001 cm) to discretize the potential distribution.


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
    V_t = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the built-in potential V_bi
    V_bi = V_t * np.log((N_a * N_d) / n_i**2)

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0

    # Calculate the depletion widths
    xn = np.sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_a / N_d))
    xp = np.sqrt((2 * epsilon * V_bi) / (q * (N_a + N_d)) * (N_d / N_a))

    # Total depletion width
    W = xn + xp

    # Space increment in cm (0.1 nm = 0.00001 cm)
    dx = 0.00001

    # Number of points in the potential array
    num_points = int(W / dx) + 1

    # Create the potential array
    potential = np.zeros(num_points)

    # Calculate the potential distribution
    for i in range(num_points):
        x = i * dx
        if x < xp:
            # Linear potential drop in the p-type region
            potential[i] = (V_bi / W) * x
        elif x < W:
            # Linear potential drop in the n-type region
            potential[i] = V_bi - (V_bi / W) * (x - xp)

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
