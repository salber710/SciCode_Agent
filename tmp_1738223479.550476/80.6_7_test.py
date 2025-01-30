from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the differences in coordinates
    dx, dy, dz = r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]

    # Use numpy-like approach without actual numpy for minimum image convention
    def wrap(d, L):
        return (d + L / 2) % L - L / 2

    # Adjust differences
    dx, dy, dz = wrap(dx, L), wrap(dy, L), wrap(dz, L)

    # Calculate the distance using the adjusted coordinates
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return distance


def E_ij(r, sigma, epsilon, rc):
    '''Calculate the truncated and shifted Lennard-Jones potential energy between two particles.

    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.

    Returns:
    float: The potential energy between the two particles, considering the specified potentials.
    '''
    # Check if the distance is beyond the cutoff
    if r >= rc:
        return 0.0
    
    # Use an array to store intermediate values
    factors = [sigma / r, sigma / rc]
    
    # Calculate sixth powers using a loop
    inv_powers = [factor ** 6 for factor in factors]
    
    # Calculate the Lennard-Jones potential using a different mathematical approach
    potential_r = epsilon * ((inv_powers[0] ** 2) - 3 * (inv_powers[0] ** 0.5))
    potential_rc = epsilon * ((inv_powers[1] ** 2) - 3 * (inv_powers[1] ** 0.5))
    
    # Return the truncated and shifted potential
    return 2 * (potential_r - potential_rc)



def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using a distinct nested loop approach with manual distance calculations.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Length of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The total potential energy of the system (in zeptojoules).
    '''
    
    def minimum_image_distance(r1, r2, L):
        # Calculate distance vector with manual wrap for minimum image
        d = [0.0] * 3
        for k in range(3):
            d[k] = r2[k] - r1[k]
            if d[k] > L / 2:
                d[k] -= L
            elif d[k] < -L / 2:
                d[k] += L
        return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

    def E_ij(r, sigma, epsilon, rc):
        if r >= rc:
            return 0.0
        inv_r6 = (sigma / r) ** 6
        inv_rc6 = (sigma / rc) ** 6
        potential_r = 4 * epsilon * (inv_r6 ** 2 - inv_r6)
        potential_rc = 4 * epsilon * (inv_rc6 ** 2 - inv_rc6)
        return potential_r - potential_rc

    E = 0.0
    N = len(xyz)

    # Calculate energy using nested loops with direct distance calculation
    for i in range(N - 1):
        for j in range(i + 1, N):
            r = minimum_image_distance(xyz[i], xyz[j], L)
            E += E_ij(r, sigma, epsilon, rc)

    return E



def f_ij(r_vector, sigma, epsilon, rc):
    '''Calculate the Lennard-Jones force vector between two particles using a completely distinct approach.

    Parameters:
    r_vector (array_like): The displacement vector between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.

    Returns:
    array_like: The force vector experienced by particle i due to particle j.
    '''
    # Compute the squared distance using a direct method
    r_squared = np.sum(r_vector ** 2)
    
    # Return zero force vector if the squared distance is beyond the squared cutoff
    if r_squared >= rc ** 2:
        return np.zeros_like(r_vector)
    
    # Calculate the inverse sixth power by expressing directly in terms of r_squared
    inv_r6 = (sigma ** 2 / r_squared) ** 3
    
    # Calculate terms for computing force magnitude using a combination of powers
    inv_r12 = inv_r6 * inv_r6
    factor = 24 * epsilon / r_squared
    
    # Utilize a new combination of terms for force magnitude
    force_magnitude = factor * (4 * inv_r12 - inv_r6)
    
    # Scale the force magnitude with the displacement vector to get the force vector
    force_vector = force_magnitude * r_vector
    
    return force_vector



def forces(N, xyz, L, sigma, epsilon, rc):
    '''Calculate the net forces acting on each particle in a system due to all pairwise interactions.
    Parameters:
    N : int
        The number of particles in the system.
    xyz : ndarray
        A NumPy array with shape (N, 3) containing the positions of each particle in the system,
        in nanometers.
    L : float
        The length of the side of the cubic simulation box (in nanometers), used for applying the minimum
        image convention in periodic boundary conditions.
    sigma : float
        The Lennard-Jones size parameter (in nanometers), indicating the distance at which the
        inter-particle potential is zero.
    epsilon : float
        The depth of the potential well (in zeptojoules), indicating the strength of the particle interactions.
    rc : float
        The cutoff distance (in nanometers) beyond which the inter-particle forces are considered negligible.
    Returns:
    ndarray
        A NumPy array of shape (N, 3) containing the net force vectors acting on each particle in the system,
        in zeptojoules per nanometer (zJ/nm).
    '''

    def apply_minimum_image(d, L):
        # Apply minimum image convention for periodic boundary conditions
        return np.mod(d + 0.5 * L, L) - 0.5 * L

    def lennard_jones_force(distance_vector, sigma, epsilon, rc):
        r2 = np.sum(distance_vector ** 2)
        if r2 >= rc ** 2:
            return np.zeros_like(distance_vector)
        
        sr2 = (sigma ** 2) / r2
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2
        force_magnitude = 48 * epsilon * (sr12 - 0.5 * sr6) / r2
        return force_magnitude * distance_vector

    forces_output = np.zeros((N, 3))

    for i in range(N):
        for j in range(i + 1, N):
            distance_vector = apply_minimum_image(xyz[j] - xyz[i], L)
            force_vector = lennard_jones_force(distance_vector, sigma, epsilon, rc)
            forces_output[i] += force_vector
            forces_output[j] -= force_vector

    return forces_output




def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    def lj_forces(positions, L, sigma, epsilon, rc):
        forces = np.zeros_like(positions)
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[j] - positions[i]
                r_vec -= np.round(r_vec / L) * L  # Apply periodic boundary conditions
                r_sq = np.dot(r_vec, r_vec)

                if r_sq < rc ** 2:
                    inv_r2 = (sigma**2) / r_sq
                    inv_r6 = inv_r2**3
                    inv_r12 = inv_r6**2
                    f_mag = 48 * epsilon * (inv_r12 - 0.5 * inv_r6) / r_sq
                    f_vector = f_mag * r_vec
                    forces[i] += f_vector
                    forces[j] -= f_vector
        return forces

    # Step 1: Calculate initial forces
    forces_initial = lj_forces(positions, L, sigma, epsilon, rc)

    # Step 2: Update positions using the initial velocities and half of the initial forces
    half_dt2 = 0.5 * dt**2
    new_positions = positions + velocities * dt + forces_initial * (half_dt2 / m)

    # Step 3: Calculate forces at the new positions
    forces_new = lj_forces(new_positions, L, sigma, epsilon, rc)

    # Step 4: Update velocities using average of initial and new forces
    new_velocities = velocities + ((forces_initial + forces_new) * 0.5 * dt / m)

    return new_positions, new_velocities


try:
    targets = process_hdf5_to_tuple('80.6', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    N = 2
    positions =np.array([[5.53189438, 7.24107158, 6.12400066],
     [5.48717536, 4.31571988, 6.51183]])  # N particles in a 10x10x10 box
    velocities = np.array([[4.37599538, 8.91785328, 9.63675087],
     [3.83429192, 7.91712711, 5.28882593]])  # N particles in a 10x10x10 box
    L = 10.0
    sigma = 1.0
    epsilon = 1.0
    m = 1
    dt = 0.01
    rc = 5
    assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    N = 5
    positions = np.array([[4.23726683, 7.24497545, 0.05701276],
     [3.03736449, 1.48736913, 1.00346047],
     [1.9594282,  3.48694961, 4.03690693],
     [5.47580623, 4.28140578, 6.8606994 ],
     [2.04842798, 8.79815741, 0.36169018]])  # N particles in a 10x10x10 box
    velocities = np.array([[6.70468142, 4.17305434, 5.5869046 ],
     [1.40388391, 1.98102942, 8.00746021],
     [9.68260045, 3.13422647, 6.92321085],
     [8.76388599, 8.9460611,  0.85043658],
     [0.39054783, 1.6983042,  8.78142503]])  # N particles in a 10x10x10 box
    L = 10.0
    sigma = 1.0
    epsilon = 1.0
    m = 1
    dt = 0.01
    rc = 5
    assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    N = 10
    positions = np.array([[4.4067278,  0.27943667, 5.56066548],
     [4.40153135, 4.25420214, 3.34203792],
     [2.12585237, 6.25071237, 3.01277888],
     [2.73834532, 6.30779077, 5.34141911],
     [1.43475136, 5.16994248, 1.90111297],
     [7.89610915, 8.58343073, 5.02002737],
     [8.51917219, 0.89182592, 5.10687864],
     [0.66107906, 4.31786204, 1.05039873],
     [1.31222278, 5.97016888, 2.28483329],
     [1.0761712,  2.30244719, 3.5953208 ]])  # N particles in a 10x10x10 box
    velocities = np.array([[4.67788095, 2.01743837, 6.40407336],
     [4.83078905, 5.0524579,  3.86901721],
     [7.93680657, 5.80047382, 1.62341802],
     [7.00701382, 9.64500116, 4.99957397],
     [8.89518145, 3.41611733, 5.67142208],
     [4.27603192, 4.36804492, 7.76616414],
     [5.35546945, 9.53684998, 5.44150931],
     [0.8219327,  3.66440749, 8.50948852],
     [4.06178341, 0.27105663, 2.47080537],
     [0.67142725, 9.93850366, 9.70578668]])  # N particles in a 10x10x10 box
    L = 10.0
    sigma = 1.0
    epsilon = 1.0
    m = 1
    dt = 0.01
    rc = 5
    assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e