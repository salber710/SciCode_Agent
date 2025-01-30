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
    def lennard_jones_forces(positions, L, sigma, epsilon, rc):
        forces = np.zeros((N, 3))
        for i in range(N):
            for j in range(i + 1, N):
                r = positions[j] - positions[i]
                r -= np.rint(r / L) * L  # Apply periodic boundary conditions
                r_sq = np.dot(r, r)

                if r_sq < rc ** 2:
                    inv_r2 = sigma ** 2 / r_sq
                    inv_r6 = inv_r2 ** 3
                    inv_r12 = inv_r6 ** 2
                    force_mag = 48 * epsilon * (inv_r12 - 0.5 * inv_r6) / r_sq
                    force = force_mag * r
                    forces[i] += force
                    forces[j] -= force
        return forces

    # Calculate initial forces
    initial_forces = lennard_jones_forces(positions, L, sigma, epsilon, rc)

    # Update positions
    half_dt = dt * 0.5
    positions_half_step = positions + velocities * half_dt + (initial_forces / m) * (half_dt ** 2)
    
    # Calculate forces at half-step positions
    forces_half_step = lennard_jones_forces(positions_half_step, L, sigma, epsilon, rc)
    
    # Update positions to full step using half-step forces
    new_positions = positions + velocities * dt + (forces_half_step / m) * (dt ** 2)
    
    # Calculate new forces at full-step positions
    new_forces = lennard_jones_forces(new_positions, L, sigma, epsilon, rc)
    
    # Update velocities
    new_velocities = velocities + (initial_forces + new_forces) / (2 * m) * dt

    return new_positions, new_velocities





def MD_NVT(N, init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu):
    '''Integrate the equations of motion using the velocity Verlet algorithm, with the inclusion of the Anderson thermostat
    for temperature control.
    
    Parameters:
    N: int
       The number of particles in the system.
    init_positions: 2D array of floats with shape (N,3)
              current positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate, units: nanometers.
    init_velocities: 2D array of floats with shape (N,3)
              current velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate, units: nanometers.
    L: float
        Length of the cubic simulation box's side, units: nanometers.
    sigma: float
       the distance at which Lennard Jones potential reaches zero, units: nanometers.
    epsilon: float
             potential well depth of Lennard Jones potential, units: zeptojoules.
    rc: float
        Cutoff radius for potential calculation, units: nanometers.
    m: float
       Mass of each particle, units: grams/mole.
    dt: float
        Integration timestep, units: picoseconds
    num_steps: float
        step number
    T: float
      Current temperature of the particles
    nu: float
      Frequency of the collision

    Returns:
    tuple
        Updated positions and velocities of all particles, and the possibly modified box length after barostat adjustment.
    '''
    # Constants
    k_B = sp.Boltzmann * 1e-21  # Boltzmann constant in zeptojoules/K
    Avogadro_number = sp.Avogadro  # Avogadro's number

    # Initialization
    positions = np.array(init_positions, copy=True)
    velocities = np.array(init_velocities, copy=True)
    E_total_array = []
    instant_T_array = []
    intercollision_times = []

    def compute_lj_forces(positions):
        forces = np.zeros_like(positions)
        for i in range(N):
            for j in range(i + 1, N):
                r = positions[j] - positions[i]
                r -= np.floor((r + L/2) / L) * L  # Periodic boundary conditions
                r_sq = np.dot(r, r)

                if r_sq < rc ** 2:
                    inv_r2 = (sigma ** 2) / r_sq
                    inv_r6 = inv_r2 ** 3
                    inv_r12 = inv_r6 ** 2
                    force_mag = 4 * epsilon * (12 * inv_r12 - 6 * inv_r6) / r_sq
                    force = force_mag * r
                    forces[i] += force
                    forces[j] -= force
        return forces

    def compute_temperature(velocities):
        kinetic_energy = 0.5 * m * np.sum(velocities**2) / Avogadro_number
        return (2 * kinetic_energy) / (3 * N * k_B)

    # Main MD loop
    for step in range(num_steps):
        # Calculate forces using Lennard-Jones potential
        forces = compute_lj_forces(positions)

        # Velocity Verlet integration
        positions += velocities * dt + (forces / m) * (dt**2) * 0.5
        new_forces = compute_lj_forces(positions)
        velocities += (forces + new_forces) * (dt / (2 * m))

        # Anderson thermostat: apply random velocity reassignment
        collision_time = step * dt
        if np.random.uniform(0, 1) < nu * dt:
            stddev = np.sqrt(k_B * T / m)
            velocities = np.random.normal(0, stddev, velocities.shape)
            intercollision_times.append(collision_time)

        # Compute instantaneous temperature
        instant_T = compute_temperature(velocities)
        instant_T_array.append(instant_T)

        # Compute total energy (kinetic + potential)
        kinetic_energy = 0.5 * m * np.sum(velocities**2) / Avogadro_number
        potential_energy = 0.5 * np.sum(forces * positions)
        total_energy = kinetic_energy + potential_energy
        E_total_array.append(total_energy)

    return E_total_array, instant_T_array, positions, velocities, intercollision_times


try:
    targets = process_hdf5_to_tuple('80.7', 3)
    target = targets[0]
    import itertools
    def initialize_fcc(N,spacing = 1.3):
        ## this follows HOOMD tutorial ##
        K = int(np.ceil(N ** (1 / 3)))
        L = K * spacing
        x = np.linspace(-L / 2, L / 2, K, endpoint=False)
        position = list(itertools.product(x, repeat=3))
        return [np.array(position),L]
    m = 1
    sigma = 1
    epsilon = 1
    dt = 0.001
    nu = 0.2/dt
    T = 1.5
    N = 200
    rc = 2.5
    num_steps = 2000
    init_positions, L = initialize_fcc(N)
    init_velocities = np.zeros(init_positions.shape)
    E_total_array,instant_T_array,positions,velocities,intercollision_times = MD_NVT(N,init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu)
    T_thresh=0.05
    s1 = np.abs(np.mean(instant_T_array[int(num_steps*0.2):])-T)/T < T_thresh
    nu_thresh=0.05
    v1 = np.abs(np.mean(1/np.mean(intercollision_times))-nu)/nu < nu_thresh
    assert (s1, v1) == target

    target = targets[1]
    import itertools
    def initialize_fcc(N,spacing = 1.3):
        ## this follows HOOMD tutorial ##
        K = int(np.ceil(N ** (1 / 3)))
        L = K * spacing
        x = np.linspace(-L / 2, L / 2, K, endpoint=False)
        position = list(itertools.product(x, repeat=3))
        return [np.array(position),L]
    m = 1
    sigma = 1
    epsilon = 1
    dt = 0.0001
    nu = 0.2/dt
    T = 100
    N = 200
    rc = 2.5
    num_steps = 2000
    init_positions, L = initialize_fcc(N)
    init_velocities = np.zeros(init_positions.shape)
    E_total_array,instant_T_array,positions,velocities,intercollision_times = MD_NVT(N,init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu)
    T_thresh=0.05
    s1 = np.abs(np.mean(instant_T_array[int(num_steps*0.2):])-T)/T < T_thresh
    nu_thresh=0.05
    v1 = np.abs(np.mean(1/np.mean(intercollision_times))-nu)/nu < nu_thresh
    assert (s1, v1) == target

    target = targets[2]
    import itertools
    def initialize_fcc(N,spacing = 1.3):
        ## this follows HOOMD tutorial ##
        K = int(np.ceil(N ** (1 / 3)))
        L = K * spacing
        x = np.linspace(-L / 2, L / 2, K, endpoint=False)
        position = list(itertools.product(x, repeat=3))
        return [np.array(position),L]
    m = 1
    sigma = 1
    epsilon = 1
    dt = 0.001
    nu = 0.2/dt
    T = 0.01
    N = 200
    rc = 2
    num_steps = 2000
    init_positions, L = initialize_fcc(N)
    init_velocities = np.zeros(init_positions.shape)
    E_total_array,instant_T_array,positions,velocities,intercollision_times = MD_NVT(N,init_positions, init_velocities, L, sigma, epsilon, rc, m, dt, num_steps, T, nu)
    T_thresh=0.05
    s1 = np.abs(np.mean(instant_T_array[int(num_steps*0.2):])-T)/T < T_thresh
    nu_thresh=0.05
    v1 = np.abs(np.mean(1/np.mean(intercollision_times))-nu)/nu < nu_thresh
    assert (s1, v1) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e