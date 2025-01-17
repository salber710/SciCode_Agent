import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro
def wrap(r, L):
    """Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    """
    coord = np.mod(r, L)
    return coord
def dist(r1, r2, L):
    """Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    """
    delta = np.array(r2) - np.array(r1)
    delta = delta - L * np.round(delta / L)
    distance = np.linalg.norm(delta)
    return distance
def dist_v(r1, r2, L):
    """Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    """
    delta = np.array(r2) - np.array(r1)
    delta = delta - L * np.round(delta / L)
    return delta
def E_ij(r, sigma, epsilon, rc):
    """Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The combined potential energy between the two particles, considering the specified potentials.
    """
    if r < rc:
        V_r = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        V_rc = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        return V_r - V_rc
    else:
        return 0.0
def f_ij(r, sigma, epsilon, rc):
    """Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    """
    if r < rc:
        force_magnitude = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
        return force_magnitude
    else:
        return 0.0
def E_tail(N, L, sigma, epsilon, rc):
    """Calculate the energy tail correction for a system of particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Length of cubic box.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    """
    V = L ** 3
    rho = N / V
    E_tail_LJ = 8 / 3 * math.pi * N * (N - 1) * rho * epsilon * ((sigma / rc) ** 9 - 3 * (sigma / rc) ** 3)
    E_tail_LJ *= 1e+21
    return E_tail_LJ

# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff distance
# to reduce computational cost. However, this truncation can lead to inaccuracies in calculated properties such as
# pressure. To account for the effects of truncation, tail corrections are applied. The pressure tail correction
# accounts for the contribution of the interactions beyond the cutoff distance. This correction is derived from
# integrating the Lennard-Jones potential over the volume outside the cutoff sphere. The correction depends on the
# number density of the particles, the cutoff distance, and the parameters of the Lennard-Jones potential (sigma and
# epsilon). The pressure tail correction is typically expressed in units of pressure, such as bar.





def P_tail(N, L, sigma, epsilon, rc):
    ''' Calculate the pressure tail correction for a system of particles, including
     the truncated and shifted Lennard-Jones contributions.
    Parameters:
     N (int): The total number of particles in the system.
     L (float): Length of cubic box
     sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
     epsilon (float): The depth of the potential well for the Lennard-Jones potential.
     rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
     Returns:
     float
         The pressure tail correction for the entire system (in bar).
    '''
    V = L ** 3
    rho = N / V
    P_tail_LJ = (16 / 3) * math.pi * rho**2 * epsilon * ((2/3) * (sigma / rc)**9 - (sigma / rc)**3)
    P_tail_LJ *= 1e-5 * Avogadro  # Convert from Pa to bar using Avogadro's number
    return P_tail_LJ


# Background: In molecular dynamics simulations, the potential energy of a system of particles is a crucial quantity
# that determines the system's stability and behavior. The Lennard-Jones potential is commonly used to model the
# interactions between particles. It accounts for the attractive and repulsive forces between particles based on
# their distance. The potential energy of a system is calculated by summing the pairwise Lennard-Jones potentials
# between all particles, considering periodic boundary conditions and a cutoff distance to truncate and shift the
# potential to zero beyond this distance. This ensures computational efficiency while maintaining accuracy.





def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.
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
    N = xyz.shape[0]
    E = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r1 = xyz[i]
            r2 = xyz[j]
            # Calculate the minimum image distance
            delta = r2 - r1
            delta = delta - L * np.round(delta / L)
            r = np.linalg.norm(delta)
            
            # Calculate the Lennard-Jones potential energy for this pair
            if r < rc:
                V_r = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
                V_rc = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
                E += V_r - V_rc

    # Convert energy to zeptojoules
    E *= 1e+21
    return E


# Background: In molecular dynamics simulations, the temperature of a system can be calculated using the equipartition theorem, which relates the kinetic energy of the particles to the temperature. The kinetic energy of a system of particles is given by the sum of the kinetic energies of all individual particles. For a system of N particles, each with mass m and velocity v, the total kinetic energy (KE) is (1/2) * m * sum(v_i^2) for all particles i. According to the equipartition theorem, the average kinetic energy per degree of freedom is (1/2) * k_B * T, where k_B is the Boltzmann constant and T is the temperature. In three dimensions, each particle has three degrees of freedom, so the total kinetic energy is (3/2) * N * k_B * T. Solving for T gives T = (2/3) * KE / (N * k_B). The Boltzmann constant k_B is 0.0138064852 zJ/K.

def temperature(v_xyz, m, N):
    '''Calculate the instantaneous temperature of a system of particles using the equipartition theorem.
    Parameters:
    v_xyz : ndarray
        A NumPy array with shape (N, 3) containing the velocities of each particle in the system,
        in nanometers per picosecond (nm/ps).
    m : float
        The molar mass of the particles in the system, in grams per mole (g/mol).
    N : int
        The number of particles in the system.
    Returns:
    float
        The instantaneous temperature of the system in Kelvin (K).
    '''
    # Convert molar mass from g/mol to kg
    m_kg = m / 1000 / Avogadro  # kg per particle

    # Calculate the kinetic energy
    v_squared = np.sum(v_xyz**2, axis=1)  # Sum of squares of velocities for each particle
    KE = 0.5 * m_kg * np.sum(v_squared)  # Total kinetic energy in kg*(nm/ps)^2

    # Convert kinetic energy to zeptojoules (1 kg*(nm/ps)^2 = 1e-21 J)
    KE_zJ = KE * 1e-21

    # Boltzmann constant in zeptojoules per Kelvin
    k_B = 0.0138064852

    # Calculate temperature using the equipartition theorem
    T = (2 / 3) * KE_zJ / (N * k_B)

    return T


# Background: In molecular dynamics simulations, the pressure of a system can be calculated using the virial equation.
# The virial equation relates the pressure to the kinetic energy and the virial of the forces acting on the particles.
# The kinetic pressure is derived from the ideal gas law, where P_kinetic = (N * k_B * T) / V, with V being the volume
# of the simulation box. The virial pressure accounts for the interactions between particles and is calculated using
# the forces between particles. The total pressure is the sum of the kinetic and virial pressures. The Boltzmann
# constant k_B is 0.0138064852 zJ/K, and the pressure is typically expressed in bar.

def pressure(N, L, T, xyz, sigma, epsilon, rc):
    '''Calculate the pressure of a system of particles using the virial theorem, considering
    the Lennard-Jones contributions.
    Parameters:
    N : int
        The number of particles in the system.
    L : float
        The length of the side of the cubic simulation box (in nanometers).
    T : float
        The instantaneous temperature of the system (in Kelvin).
    xyz : ndarray
        A NumPy array with shape (N, 3) containing the positions of each particle in the system, in nanometers.
    sigma : float
        The Lennard-Jones size parameter (in nanometers).
    epsilon : float
        The depth of the potential well (in zeptojoules).
    rc : float
        The cutoff distance beyond which the inter-particle potential is considered to be zero (in nanometers).
    Returns:
    tuple
        The kinetic pressure (in bar), the virial pressure (in bar), and the total pressure (kinetic plus virial, in bar) of the system.
    '''
    # Calculate the volume of the cubic box
    V = L ** 3

    # Boltzmann constant in zeptojoules per Kelvin
    k_B = 0.0138064852

    # Calculate kinetic pressure using the ideal gas law
    P_kinetic = (N * k_B * T) / V

    # Initialize virial sum
    virial_sum = 0.0

    # Calculate the virial contribution from pairwise interactions
    for i in range(N):
        for j in range(i + 1, N):
            r1 = xyz[i]
            r2 = xyz[j]
            # Calculate the minimum image vector
            delta = r2 - r1
            delta = delta - L * np.round(delta / L)
            r = np.linalg.norm(delta)

            # Calculate the force magnitude if within cutoff
            if r < rc:
                force_magnitude = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
                virial_sum += np.dot(delta, delta) * force_magnitude

    # Calculate virial pressure
    P_virial = virial_sum / (3 * V)

    # Convert pressures from zeptojoules per nanometer cubed to bar
    P_kinetic *= 1e-5 * Avogadro
    P_virial *= 1e-5 * Avogadro

    return P_kinetic, P_virial, P_kinetic + P_virial


# Background: In molecular dynamics simulations, calculating the forces on each particle due to interactions with
# other particles is crucial for understanding the system's dynamics. The Lennard-Jones potential is often used to
# model these interactions, which include both attractive and repulsive components. The force between two particles
# can be derived from the Lennard-Jones potential, and it is important to consider only interactions within a cutoff
# distance to ensure computational efficiency. The net force on each particle is the vector sum of the forces due to
# all other particles, considering periodic boundary conditions to account for the finite size of the simulation box.





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
    # Initialize the force array
    f_xyz = np.zeros((N, 3))

    # Loop over all pairs of particles
    for i in range(N):
        for j in range(i + 1, N):
            r1 = xyz[i]
            r2 = xyz[j]
            # Calculate the minimum image vector
            delta = r2 - r1
            delta = delta - L * np.round(delta / L)
            r = np.linalg.norm(delta)

            # Calculate the force magnitude if within cutoff
            if r < rc:
                force_magnitude = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
                force_vector = force_magnitude * (delta / r)
                # Update the forces on particles i and j
                f_xyz[i] += force_vector
                f_xyz[j] -= force_vector

    return f_xyz



# Background: The Berendsen thermostat and barostat are methods used in molecular dynamics simulations to control the
# temperature and pressure of the system, respectively. The Berendsen thermostat rescales the velocities of particles
# to bring the system's temperature closer to a target temperature, T_target, over a characteristic time, tau_T. The
# Berendsen barostat rescales the simulation box and particle positions to adjust the system's pressure towards a
# target pressure, P_target, over a characteristic time, tau_P. The velocity Verlet algorithm is a common method for
# integrating the equations of motion in molecular dynamics, providing a way to update positions and velocities of
# particles based on forces. The integration of the Berendsen thermostat and barostat into the velocity Verlet
# algorithm allows for the control of temperature and pressure during the simulation, ensuring the system remains
# at desired conditions.





def velocityVerlet(N, xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target):
    '''Integrate the equations of motion using the velocity Verlet algorithm, with the inclusion of the Berendsen thermostat
    and barostat for temperature and pressure control, respectively.
    Parameters:
    N : int
        The number of particles in the system.
    xyz : ndarray
        Current particle positions in the system, shape (N, 3), units: nanometers.
    v_xyz : ndarray
        Current particle velocities in the system, shape (N, 3), units: nanometers/ps.
    L : float
        Length of the cubic simulation box's side, units: nanometers.
    sigma : float
        Lennard-Jones potential size parameter, units: nanometers.
    epsilon : float
        Lennard-Jones potential depth parameter, units: zeptojoules.
    rc : float
        Cutoff radius for potential calculation, units: nanometers.
    m : float
        Mass of each particle, units: grams/mole.
    dt : float
        Integration timestep, units: picoseconds.
    tau_T : float
        Temperature coupling time constant for the Berendsen thermostat. Set to 0 to deactivate, units: picoseconds.
    T_target : float
        Target temperature for the Berendsen thermostat, units: Kelvin.
    tau_P : float
        Pressure coupling time constant for the Berendsen barostat. Set to 0 to deactivate, units: picoseconds.
    P_target : float
        Target pressure for the Berendsen barostat, units: bar.
    Returns:
    --------
    xyz_full : ndarray
        Updated particle positions in the system, shape (N, 3), units: nanometers.
    v_xyz_full : ndarray
        Updated particle velocities in the system, shape (N, 3), units: nanometers/ps.
    L : float
        Updated length of the cubic simulation box's side, units: nanometers.
    Raises:
    -------
    Exception:
        If the Berendsen barostat has shrunk the box such that the side length L is less than twice the cutoff radius.
    '''

    # Calculate initial forces
    f_xyz = forces(N, xyz, L, sigma, epsilon, rc)

    # Update positions
    xyz_full = xyz + v_xyz * dt + 0.5 * f_xyz * (dt**2) / (m / 1000 / Avogadro)

    # Apply periodic boundary conditions
    xyz_full = np.mod(xyz_full, L)

    # Calculate new forces
    f_xyz_new = forces(N, xyz_full, L, sigma, epsilon, rc)

    # Update velocities
    v_xyz_full = v_xyz + 0.5 * (f_xyz + f_xyz_new) * dt / (m / 1000 / Avogadro)

    # Apply Berendsen thermostat
    if tau_T > 0:
        T_current = temperature(v_xyz_full, m, N)
        lambda_T = np.sqrt(1 + (dt / tau_T) * ((T_target / T_current) - 1))
        v_xyz_full *= lambda_T

    # Apply Berendsen barostat
    if tau_P > 0:
        P_kinetic, P_virial, P_current = pressure(N, L, T_target, xyz_full, sigma, epsilon, rc)
        lambda_P = 1 - (dt / tau_P) * (P_current - P_target) / (3 * P_current)
        L *= lambda_P
        xyz_full *= lambda_P

        # Check if the box has shrunk too much
        if L < 2 * rc:
            raise Exception("The Berendsen barostat has shrunk the box such that the side length L is less than twice the cutoff radius.")

    return xyz_full, v_xyz_full, L


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('77.12', 1)
target = targets[0]

np.random.seed(17896)
# NPT simulation
T_target = 298 # K
P_target = 200 # bar
L = 2.4 # nm
N = 100
dt = 0.005 # ps
nSteps = 1200
rc = 0.8 # nm
printModulus = 1 # steps
sigma = 0.34 # nm
epsilon = 1.65 # zJ
tau_T = 0.1 # ps
tau_P = 0.01 # ps
kB = 1.38064852E-2 # zJ/K
m = 39.948 # g/mol
gamma = 4.6E-5 # 1/bar (isothermal compressibility of water at 1 bar and 300 K)
# position initialization -- random
def init_rand(N,L,sigma):
  """
    Initialize the positions of N particles randomly within a cubic box of side length L,
    ensuring that no two particles are closer than a distance of sigma.
    Parameters:
    -----------
    N : int
        Number of particles to initialize.
    L : float
        Length of each side of the cubic box.
    sigma : float
        Minimum allowed distance between any two particles.
    Returns:
    --------
    xyz : ndarray
        Array of shape (N, 3) containing the initialized positions of the particles.
        Be sure to use np.random.uniform to initialize it.
    Raises:
    -------
    Exception
        If a collision is detected after initialization.
    """
  xyz = np.random.uniform(0,L,(N,3))
  for ii in range(N):
      #print('  Inserting particle %d' % (ii+1))
      xyz[ii,:] = np.random.uniform(0,L,3)
      r1 = xyz[ii,:]
      collision=1
      while(collision):
          collision=0
          for jj in range(ii):
              r2 = xyz[jj,:]
              d = dist(r1,r2,L)
              if d<sigma:
                  collision=1
                  break
          if collision:
              r1 = np.random.uniform(0,L,3)
              xyz[ii,:] = r1
  # verifying all collisions resolved
  for ii in range(N):
      r1 = xyz[ii,:]
      for jj in range(ii):
          r2 = xyz[jj,:]
          d = dist(r1,r2,L)
          if d<sigma:
              raise Exception('Collision between particles %d and %d' % (ii+1,jj+1))
  return xyz
def vMaxBoltz(T, N, m):
    """
    Initialize velocities of particles according to the Maxwell-Boltzmann distribution.
    Parameters:
    -----------
    T : float
        Temperature in Kelvin.
    N : int
        Number of particles.
    m : float
        Molecular mass of the particles in grams per mole (g/mol).
    Returns:
    --------
    v_xyz : ndarray
        Array of shape (N, 3) containing the initialized velocities of the particles
        in nm/ps.
    """
    kB = 1.38064852E-2 # zJ/K
    kB_J_per_K = kB * 1e-21
    m_kg_per_particle = m * 1e-3 / Avogadro  # Convert g/mol to kg/particle
    std_dev_m_per_s = np.sqrt(kB_J_per_K * T / m_kg_per_particle)  # Standard deviation for the velocity components in m/s
    # Initialize velocities from a normal distribution in nm/ps
    v_xyz = np.random.normal(0, std_dev_m_per_s, (N, 3)) * 1e-3
    # Subtract the center of mass velocity in each dimension to remove net momentum
    v_cm = np.mean(v_xyz, axis=0)
    v_xyz -= v_cm
    return v_xyz  # v_xyz in nm/ps
def test_main(N, xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target, nSteps):
  """
    Simulate molecular dynamics using the Velocity-Verlet algorithm and observe properties
    such as temperature and pressure over a specified number of steps.
    Parameters:
    -----------
    N : int
        Number of particles in the system.
    xyz : ndarray
        Positions of the particles in the system with shape (N, 3).
    v_xyz : ndarray
        Velocities of the particles in the system with shape (N, 3).
    L : float
        Length of the cubic box.
    sigma : float
        Distance parameter for the Lennard-Jones potential.
    epsilon : float
        Depth of the potential well for the Lennard-Jones potential.
    rc : float
        Cutoff radius for the potential.
    m : float
        Mass of a single particle.
    dt : float
        Time step for the simulation.
    tau_T : float
        Relaxation time for the temperature coupling.
    T_target : float
        Target temperature for the system.
    tau_P : float
        Relaxation time for the pressure coupling.
    P_target : float
        Target pressure for the system.
    nSteps : int
        Number of simulation steps to be performed.
    Returns:
    --------
    T_traj : ndarray
        Trajectory of the temperature over the simulation steps.
    P_traj : ndarray
        Trajectory of the pressure over the simulation steps.
  """
  T_traj = []
  P_traj = []
  for step in range(nSteps):
      xyz, v_xyz, L = velocityVerlet(N, xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target)
      if (step+1) % printModulus == 0:
          T = temperature(v_xyz,m,N)
          P_kin, P_vir, P = pressure(N,L,T,xyz,sigma,epsilon,rc)
          T_traj.append(T)
          P_traj.append(P)
  T_traj = np.array(T_traj)
  P_traj = np.array(P_traj)
  return  T_traj, P_traj
# initializing atomic positions and velocities and writing to file
xyz = init_rand(N,L,sigma)
# initializing atomic velocities and writing to file
v_xyz = vMaxBoltz(T_target,N,m)
T_sim, P_sim = test_main(N, xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target, nSteps)
threshold = 0.3
assert (np.abs(np.mean(T_sim-T_target)/T_target)<threshold and np.abs(np.mean(P_sim[int(0.2*nSteps):]-P_target)/P_target)<threshold) == target
