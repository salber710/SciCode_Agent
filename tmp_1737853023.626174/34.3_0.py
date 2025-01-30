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

    # Validate inputs
    if N_d <= 0 or N_a <= 0:
        raise ValueError("Doping concentrations (N_d, N_a) must be positive.")
    if n_i <= 0:
        raise ValueError("Intrinsic carrier density (n_i) must be positive.")
    if e_r <= 0:
        raise ValueError("Relative permittivity (e_r) must be positive.")

    # Calculate the built-in potentials
    phi_p = V_t * np.log(N_a / n_i)
    phi_n = V_t * np.log(N_d / n_i)

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0

    # Calculate the depletion widths
    xn = np.sqrt((2 * epsilon * phi_p) / (q * (N_a + N_d) * N_d))
    xp = np.sqrt((2 * epsilon * phi_n) / (q * (N_a + N_d) * N_a))

    # Handle potential numerical issues with extremely high intrinsic carrier density
    if np.isnan(xn) or np.isnan(xp):
        xn = 0  # Set to zero if numerical instability occurs
        xp = 0  # Set to zero if numerical instability occurs

    return xn, xp



# Background: In semiconductor physics, the potential distribution across a p-n junction can be visualized as a band diagram.
# The conduction band potential is typically set to 0V at the start of the depletion region on the p-type side.
# The potential changes across the depletion region due to the built-in electric field, which is a result of the charge separation.
# The depletion widths (x_n and x_p) determine the extent of the depletion region on the n-type and p-type sides, respectively.
# To visualize the potential distribution, we can create an array representing the conduction band potential across the junction.
# The potential is constant outside the depletion region and changes linearly within it due to the electric field.
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
    V_t = 0.0259  # Thermal potential at room temperature in volts

    # Calculate the built-in potentials
    phi_p = V_t * np.log(N_a / n_i)
    phi_n = V_t * np.log(N_d / n_i)

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0

    # Calculate the depletion widths
    xn = np.sqrt((2 * epsilon * phi_p) / (q * (N_a + N_d) * N_d))
    xp = np.sqrt((2 * epsilon * phi_n) / (q * (N_a + N_d) * N_a))

    # Total depletion width
    W = xn + xp

    # Discretize the potential distribution
    dx = 0.00001  # 0.1 nm in cm
    num_points = int(W / dx) + 1
    potential = np.zeros(num_points)

    # Calculate potential distribution
    for i in range(num_points):
        x = i * dx
        if x < xp:
            # In the p-type depletion region
            potential[i] = -phi_p * (x / xp)
        elif x < W:
            # In the n-type depletion region
            potential[i] = phi_n * ((x - xp) / xn)
        else:
            # Outside the depletion region
            potential[i] = 0

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
