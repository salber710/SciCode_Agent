import numpy as np

# Background: Mannella's leapfrog method is a numerical technique used to solve stochastic differential equations like the Langevin equation. 
# The Langevin equation describes the motion of a particle in a fluid, accounting for both deterministic forces (like a harmonic potential) 
# and stochastic forces (like thermal noise). In the context of an optically trapped microsphere, the equation includes terms for the 
# harmonic potential (characterized by the resonant frequency omega0) and a stochastic force related to the thermal motion of the particle 
# (characterized by the root mean square velocity vrms). The leapfrog method is particularly suitable for this type of problem because it 
# is symplectic, meaning it conserves the Hamiltonian structure of the system, which is important for accurately simulating the dynamics 
# over long times. The method involves updating positions and velocities in a staggered manner, which helps maintain stability and accuracy.


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
    
    # Validate input types
    if not all(isinstance(arg, (int, float)) for arg in [x0, v0, t0, taup, omega0, vrms]):
        raise TypeError("All parameters except 'steps' must be of type int or float.")
    if not isinstance(steps, int):
        raise TypeError("'steps' must be of type int.")
    
    # Validate input values
    if t0 < 0:
        raise ValueError("Total simulation time 't0' must be non-negative.")
    if steps <= 0:
        raise ValueError("Number of integration steps 'steps' must be greater than zero.")
    if taup <= 0:
        raise ValueError("Momentum relaxation time 'taup' must be greater than zero.")
    
    # Time step
    dt = t0 / steps
    
    # Initialize position and velocity
    x = x0
    v = v0
    
    # Precompute constants
    gamma = 1.0 / taup
    sqrt_dt = np.sqrt(dt)
    
    # Leapfrog integration
    for _ in range(steps):
        # Calculate the random force term
        random_force = vrms * np.sqrt(2 * gamma) * np.random.normal(0, 1)
        
        # Update velocity (half step)
        v += (-gamma * v - omega0**2 * x) * dt / 2 + random_force * sqrt_dt
        
        # Update position (full step)
        x += v * dt
        
        # Update velocity (half step)
        v += (-gamma * v - omega0**2 * x) * dt / 2 + random_force * sqrt_dt
    
    return x



# Background: The mean-square displacement (MSD) is a measure of the average squared distance a particle moves from its initial position over time. 
# In the context of an optically trapped microsphere in a gas, the MSD can provide insights into the dynamics of the particle under the influence 
# of both deterministic and stochastic forces. Mannella's leapfrog method is used to simulate the Langevin dynamics of the microsphere, 
# accounting for the harmonic potential and thermal noise. By averaging the results of multiple simulations (Navg), we can obtain a reliable 
# estimate of the MSD at a given time point t0. The initial position and velocity of the microsphere are sampled from the Maxwell-Boltzmann 
# distribution, which describes the distribution of speeds in a gas in thermal equilibrium.


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
    
    # Validate input types
    if not all(isinstance(arg, (int, float)) for arg in [t0, taup, omega0, vrms]):
        raise TypeError("Parameters 't0', 'taup', 'omega0', and 'vrms' must be of type int or float.")
    if not isinstance(steps, int) or not isinstance(Navg, int):
        raise TypeError("'steps' and 'Navg' must be of type int.")
    
    # Validate input values
    if t0 < 0:
        raise ValueError("Time point 't0' must be non-negative.")
    if steps <= 0:
        raise ValueError("Number of integration steps 'steps' must be greater than zero.")
    if taup <= 0:
        raise ValueError("Momentum relaxation time 'taup' must be greater than zero.")
    if Navg <= 0:
        raise ValueError("Number of simulations 'Navg' must be greater than zero.")
    
    # Time step
    dt = t0 / steps
    
    # Precompute constants
    gamma = 1.0 / taup
    sqrt_dt = np.sqrt(dt)
    
    # Initialize MSD accumulator
    msd_accumulator = 0.0
    
    for _ in range(Navg):
        # Sample initial position and velocity from Maxwell-Boltzmann distribution
        x0 = np.random.normal(0, vrms / omega0)
        v0 = np.random.normal(0, vrms)
        
        # Initialize position and velocity
        x = x0
        v = v0
        
        # Leapfrog integration
        for _ in range(steps):
            # Calculate the random force term
            random_force = vrms * np.sqrt(2 * gamma) * np.random.normal(0, 1)
            
            # Update velocity (half step)
            v += (-gamma * v - omega0**2 * x) * dt / 2 + random_force * sqrt_dt
            
            # Update position (full step)
            x += v * dt
            
            # Update velocity (half step)
            v += (-gamma * v - omega0**2 * x) * dt / 2 + random_force * sqrt_dt
        
        # Accumulate the squared displacement
        msd_accumulator += x**2
    
    # Calculate the mean-square displacement
    x_MSD = msd_accumulator / Navg
    
    return x_MSD

from scicode.parse.parse import process_hdf5_to_tuple
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
