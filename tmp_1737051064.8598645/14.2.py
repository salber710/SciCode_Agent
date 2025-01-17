import numpy as np

# Background: The Langevin equation describes the motion of a particle in a fluid, accounting for both systematic forces and random thermal forces. 
# In the context of a microsphere in an optical trap, the Langevin equation can be used to model the dynamics of the particle under the influence of 
# a harmonic potential and stochastic forces due to collisions with gas molecules. Mannella's leapfrog method is a numerical integration technique 
# that is particularly suitable for solving stochastic differential equations like the Langevin equation. It is a variant of the leapfrog method 
# that incorporates stochastic terms, allowing for the simulation of systems with noise. The method involves updating the position and velocity 
# of the particle in a staggered manner, which helps maintain stability and accuracy over long simulations.


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
    
    # Time step
    dt = t0 / steps
    
    # Initialize position and velocity
    x = x0
    v = v0
    
    # Precompute constants
    gamma = 1.0 / taup
    sqrt_dt = np.sqrt(dt)
    sqrt_2gamma_vrms2 = np.sqrt(2 * gamma * vrms**2)
    
    # Leapfrog integration using Mannella's method
    for _ in range(steps):
        # Update velocity (half step)
        v += (-gamma * v - omega0**2 * x) * (dt / 2) + sqrt_2gamma_vrms2 * np.random.normal(0, 1) * sqrt_dt
        
        # Update position (full step)
        x += v * dt
        
        # Update velocity (half step)
        v += (-gamma * v - omega0**2 * x) * (dt / 2) + sqrt_2gamma_vrms2 * np.random.normal(0, 1) * sqrt_dt
    
    return x



# Background: The mean-square displacement (MSD) is a measure of the average squared distance a particle travels over time, 
# and is a key quantity in understanding the dynamics of particles in a medium. For a microsphere in an optical trap, 
# the MSD can provide insights into the effects of thermal fluctuations and the trapping potential. 
# To calculate the MSD using Mannella's leapfrog method, we simulate the Langevin dynamics of the microsphere multiple times 
# and average the squared displacements. The initial conditions for each simulation are drawn from the Maxwell-Boltzmann distribution, 
# which describes the distribution of velocities for particles in thermal equilibrium. The MSD is then computed as the average 
# of the squared displacements from the initial position over the specified number of simulations.


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
    
    # Time step
    dt = t0 / steps
    
    # Precompute constants
    gamma = 1.0 / taup
    sqrt_dt = np.sqrt(dt)
    sqrt_2gamma_vrms2 = np.sqrt(2 * gamma * vrms**2)
    
    # Initialize MSD accumulator
    msd_accumulator = 0.0
    
    for _ in range(Navg):
        # Initial position and velocity from Maxwell distribution
        x0 = 0.0  # Assuming initial position is zero
        v0 = np.random.normal(0, vrms)
        
        # Initialize position and velocity
        x = x0
        v = v0
        
        # Leapfrog integration using Mannella's method
        for _ in range(steps):
            # Update velocity (half step)
            v += (-gamma * v - omega0**2 * x) * (dt / 2) + sqrt_2gamma_vrms2 * np.random.normal(0, 1) * sqrt_dt
            
            # Update position (full step)
            x += v * dt
            
            # Update velocity (half step)
            v += (-gamma * v - omega0**2 * x) * (dt / 2) + sqrt_2gamma_vrms2 * np.random.normal(0, 1) * sqrt_dt
        
        # Accumulate squared displacement
        msd_accumulator += x**2
    
    # Calculate mean-square displacement
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
