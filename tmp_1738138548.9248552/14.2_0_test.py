from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms):
    dt = t0 / steps
    x = x0
    v = v0
    gamma = 1 / taup
    noise_coeff = np.sqrt(2 * gamma * vrms**2 * dt)
    
    # Initialize velocity at half step with noise
    v_half = v - 0.5 * dt * (gamma * v + omega0**2 * x) + 0.5 * noise_coeff * np.random.normal()
    
    for _ in range(steps):
        # Update position using the half-step velocity
        x += v_half * dt
        
        # Generate random noise for the full step
        noise = noise_coeff * np.random.normal()
        
        # Update velocity at full step
        v_full = v_half - dt * (gamma * v_half + omega0**2 * x) + noise
        
        # Prepare half-step velocity for the next iteration
        v_half = v_full - 0.5 * dt * (gamma * v_full + omega0**2 * x) + 0.5 * noise_coeff * np.random.normal()
    
    return x



# Background: The mean-square displacement (MSD) is a measure of the average squared distance a particle moves from its initial position over time. In the context of a microsphere in an optical trap, the MSD can be used to understand the dynamics of the particle under the influence of thermal fluctuations and the trapping potential. Mannella's leapfrog method is a numerical technique used to solve stochastic differential equations like the Langevin equation, which describes the motion of particles in a fluid. The method involves updating positions and velocities in a staggered manner to maintain stability and accuracy. To calculate the MSD, we perform multiple simulations (Navg) of the particle's trajectory, each starting with initial conditions sampled from a Maxwell-Boltzmann distribution, and average the squared displacements from the initial position.

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


    dt = t0 / steps
    gamma = 1 / taup
    noise_coeff = np.sqrt(2 * gamma * vrms**2 * dt)
    
    msd_sum = 0.0

    for _ in range(Navg):
        # Initial position and velocity from Maxwell distribution
        x0 = np.random.normal(0, np.sqrt(vrms**2 / omega0**2))
        v0 = np.random.normal(0, vrms)
        
        x = x0
        v = v0
        
        # Initialize velocity at half step with noise
        v_half = v - 0.5 * dt * (gamma * v + omega0**2 * x) + 0.5 * noise_coeff * np.random.normal()
        
        for _ in range(steps):
            # Update position using the half-step velocity
            x += v_half * dt
            
            # Generate random noise for the full step
            noise = noise_coeff * np.random.normal()
            
            # Update velocity at full step
            v_full = v_half - dt * (gamma * v_half + omega0**2 * x) + noise
            
            # Prepare half-step velocity for the next iteration
            v_half = v_full - 0.5 * dt * (gamma * v_full + omega0**2 * x) + 0.5 * noise_coeff * np.random.normal()
        
        # Accumulate the squared displacement
        msd_sum += (x - x0)**2

    # Average over all simulations
    x_MSD = msd_sum / Navg

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