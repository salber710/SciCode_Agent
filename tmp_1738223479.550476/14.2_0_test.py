from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms):
    '''Function to employ Mannella's leapfrog method to solve the Langevin equation of a microsphere optically trapped in the gas.
    Input
    x0 : float
        Initial position of the microsphere.
    v0 : float
        Initial velocity of the microsphere.
    t0 : float
        Total simulation time.
    steps : int
        Number of integration steps.
    taup : float
        Momentum relaxation time of the trapped microsphere in the gas (often referred to as the particle relaxation time).
    omega0 : float
        Resonant frequency of the harmonic potential (optical trap).
    vrms : float
        Root mean square velocity of the trapped microsphere in the gas.
    Output
    x : float
        Final position of the microsphere after the simulation time.
    '''


    # Compute the time step
    dt = t0 / steps
    gamma = 1.0 / taup
    noise_magnitude = np.sqrt(2 * gamma) * vrms * np.sqrt(dt)

    # Initialize position and velocity
    x = x0
    v = v0

    for _ in range(steps):
        # Calculate the random force
        random_force = np.random.normal(0, noise_magnitude)

        # Velocity update - full step
        v_half = v + (-gamma * v - omega0**2 * x) * dt + random_force

        # Position update - full step using the halfway velocity
        x += v_half * dt
        
        # Velocity update - complete the velocity step
        random_force = np.random.normal(0, noise_magnitude)  # Recalculate random force
        v = v_half + (-gamma * v_half - omega0**2 * x) * dt + random_force

    return x



# Background: 
# The mean-square displacement (MSD) is a measure of the average squared distance a particle travels from its initial position over time. 
# For a microsphere in an optical trap, the Langevin equation describes its motion, accounting for both deterministic forces (like the harmonic potential of the trap) 
# and stochastic forces (due to thermal fluctuations of the gas). 
# Mannella's leapfrog method is a numerical technique to integrate such stochastic differential equations. 
# The initial conditions for position and velocity are derived from the Maxwell-Boltzmann distribution, 
# which describes the distribution of speeds for particles in a thermal equilibrium at a given temperature. 
# To calculate the MSD at time `t0`, we perform multiple simulations (Navg times), each time running the leapfrog integration 
# and calculating the squared displacement, and then average these squared displacements.


def calculate_msd(t0, steps, taup, omega0, vrms, Navg):
    '''Calculate the mean-square displacement (MSD) of an optically trapped microsphere in a gas by averaging Navg simulations.
    Input:
    t0 : float
        The time point at which to calculate the MSD.
    steps : int
        Number of simulation steps for the integration.
    taup : float
        Momentum relaxation time of the microsphere.
    omega0 : float
        Resonant frequency of the optical trap.
    vrms : float
        Root mean square velocity of the thermal fluctuations.
    Navg : int
        Number of simulations to average over for computing the MSD.
    Output:
    x_MSD : float
        The computed MSD at time point `t0`.
    '''
    
    # Define the function to perform a single Mannella's leapfrog simulation
    def single_simulation(x0, v0, t0, steps, taup, omega0, vrms):
        dt = t0 / steps
        gamma = 1.0 / taup
        noise_magnitude = np.sqrt(2 * gamma) * vrms * np.sqrt(dt)

        x = x0
        v = v0

        for _ in range(steps):
            random_force = np.random.normal(0, noise_magnitude)
            v_half = v + (-gamma * v - omega0**2 * x) * dt + random_force
            x += v_half * dt
            random_force = np.random.normal(0, noise_magnitude)
            v = v_half + (-gamma * v_half - omega0**2 * x) * dt + random_force

        return x

    # Accumulate the squared displacements
    squared_displacements = []

    for _ in range(Navg):
        # Sample initial position and velocity from Maxwell-Boltzmann distribution
        x0 = np.random.normal(0, vrms)
        v0 = np.random.normal(0, vrms)
        
        # Perform a single simulation
        final_x = single_simulation(x0, v0, t0, steps, taup, omega0, vrms)
        
        # Compute the squared displacement
        squared_displacement = (final_x - x0) ** 2
        squared_displacements.append(squared_displacement)

    # Calculate the mean of squared displacements
    x_MSD = np.mean(squared_displacements)

    return x_MSD


try:
    targets = process_hdf5_to_tuple('14.2', 3)
    target = targets[0]
    def analytical_msd(t0, taup, omega0, vrms):
        """
        Analytically calculate the mean-square displacement (MSD) of an optically trapped microsphere in a gas.
        Input:
        t0 : float
            The time point at which to calculate the MSD.
        taup : float
            Momentum relaxation time of the microsphere.
        omega0 : float
            Resonant frequency of the optical trap.
        vrms : float
            Root mean square velocity of the thermal fluctuations.
        Output:
        t_MSD : float
            The computed MSD at time point `t0`.
        """
        omega1 = np.sqrt(omega0 ** 2 - 1 / (4 * taup ** 2))
        t_MSD = 2 * vrms ** 2 / (omega0 ** 2) * (1 - np.exp(-t0 / (2 * taup)) * (np.cos(omega1 * t0) + np.sin(omega1 * t0) / (2 * omega1 * (taup))))
        return t_MSD
    t0 = 5e-6
    steps = 5000 #step-size in Mannella's leapfrog method
    taup = 48.5e-6
    omega0 = 2 * np.pi * 3064
    vrms = 0.422e-3
    Navg = 4000 #simulation number
    x_MSD = calculate_msd(t0, steps, taup, omega0, vrms, Navg)
    t_MSD = analytical_msd(t0, taup, omega0, vrms)
    eta = x_MSD / t_MSD
    assert (eta>0.95 and eta<1.05) == target

    target = targets[1]
    def analytical_msd(t0, taup, omega0, vrms):
        """
        Analytically calculate the mean-square displacement (MSD) of an optically trapped microsphere in a gas.
        Input:
        t0 : float
            The time point at which to calculate the MSD.
        taup : float
            Momentum relaxation time of the microsphere.
        omega0 : float
            Resonant frequency of the optical trap.
        vrms : float
            Root mean square velocity of the thermal fluctuations.
        Output:
        t_MSD : float
            The computed MSD at time point `t0`.
        """
        omega1 = np.sqrt(omega0 ** 2 - 1 / (4 * taup ** 2))
        t_MSD = 2 * vrms ** 2 / (omega0 ** 2) * (1 - np.exp(-t0 / (2 * taup)) * (np.cos(omega1 * t0) + np.sin(omega1 * t0) / (2 * omega1 * (taup))))
        return t_MSD
    t0 = 1e-5
    steps = 5000 #step-size in Mannella's leapfrog method
    taup = 48.5e-6
    omega0 = 2 * np.pi * 3064
    vrms = 0.422e-3
    Navg = 4000 #simulation number
    x_MSD = calculate_msd(t0, steps, taup, omega0, vrms, Navg)
    t_MSD = analytical_msd(t0, taup, omega0, vrms)
    eta = x_MSD / t_MSD
    assert (eta>0.95 and eta<1.05) == target

    target = targets[2]
    def analytical_msd(t0, taup, omega0, vrms):
        """
        Analytically calculate the mean-square displacement (MSD) of an optically trapped microsphere in a gas.
        Input:
        t0 : float
            The time point at which to calculate the MSD.
        taup : float
            Momentum relaxation time of the microsphere.
        omega0 : float
            Resonant frequency of the optical trap.
        vrms : float
            Root mean square velocity of the thermal fluctuations.
        Output:
        t_MSD : float
            The computed MSD at time point `t0`.
        """
        omega1 = np.sqrt(omega0 ** 2 - 1 / (4 * taup ** 2))
        t_MSD = 2 * vrms ** 2 / (omega0 ** 2) * (1 - np.exp(-t0 / (2 * taup)) * (np.cos(omega1 * t0) + np.sin(omega1 * t0) / (2 * omega1 * (taup))))
        return t_MSD
    t0 = 1e-5
    steps = 5000 #step-size in Mannella's leapfrog method
    taup = 147.3e-6
    omega0 = 2 * np.pi * 3168
    vrms = 0.425e-3
    Navg = 4000 #simulation number
    x_MSD = calculate_msd(t0, steps, taup, omega0, vrms, Navg)
    t_MSD = analytical_msd(t0, taup, omega0, vrms)
    eta = x_MSD / t_MSD
    assert (eta>0.95 and eta<1.05) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e