from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def Fermi(N_a, N_d, n_i):
    V_T = 0.0259  # Thermal voltage at room temperature in volts

    # Using the hyperbolic sine function for a unique approach
    phi_p = V_T * log((1 + (N_a / n_i)) / (1 - (N_a / n_i))) / 2
    phi_n = V_T * log((1 + (N_d / n_i)) / (1 - (N_d / n_i))) / 2

    return phi_p, phi_n



def depletion(N_a, N_d, n_i, e_r):
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19  # Electron charge in C

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0
    
    # Calculate the built-in potential using a hyperbolic tangent model
    V_bi = 0.026 * math.tanh((N_a * N_d) / (n_i**2))
    
    # Calculate the depletion widths using a cubic root scaling
    xn = (2 * epsilon * V_bi / (q * N_d)) ** (1/3) * (N_a / (N_a + N_d))
    xp = (2 * epsilon * V_bi / (q * N_a)) ** (1/3) * (N_d / (N_a + N_d))
    
    return xn, xp



# Background: 
# In semiconductor physics, the depletion region is the area around the p-n junction where mobile charge carriers are depleted. 
# The depletion width on the n-type side (x_n) and the p-type side (x_p) can be calculated using Poisson's equation, 
# which relates the electric field and potential to the charge distribution. 
# The potential distribution across the depletion region can be visualized as a band diagram, 
# where the conduction band edge is plotted. The potential is typically set to 0V at the start of the depletion region 
# on the p-type side, and it increases across the junction. 
# The potential distribution can be discretized into small increments (e.g., 0.1 nm) to create a detailed band diagram.


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
    V_T = 0.0259  # Thermal voltage at room temperature in volts

    # Calculate the permittivity of the material
    epsilon = e_r * epsilon_0

    # Calculate the built-in potential using the Fermi function
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    V_bi = phi_p + phi_n

    # Calculate the depletion widths using the depletion function
    xn, xp = depletion(N_a, N_d, n_i, e_r)

    # Total depletion width
    W = xn + xp

    # Discretize the potential across the depletion region
    dx = 0.1e-7  # 0.1 nm in cm
    num_points = int(W / dx) + 1
    x = np.linspace(0, W, num_points)

    # Calculate the potential distribution
    potential = np.zeros(num_points)
    for i in range(num_points):
        if x[i] < xp:
            # p-type side
            potential[i] = q * N_a / (2 * epsilon) * (x[i] - xp)**2
        else:
            # n-type side
            potential[i] = V_bi - q * N_d / (2 * epsilon) * (x[i] - xp)**2

    return xn, xp, potential


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e