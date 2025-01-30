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
    # Constants
    kB = 1.38e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in Kelvin
    mass = vrms**2 * taup / (2 * kB * T)  # Calculating the mass from vrms and taup

    # Time step
    dt = t0 / steps

    # Initialize position and velocity
    x = x0
    v = v0

    # Pre-calculated constants
    gamma = 1 / taup
    sqrt_2kBT_gamma = np.sqrt(2 * kB * T * gamma)

    # Leapfrog integration loop
    for _ in range(steps):
        # Generate random numbers for Gaussian noise
        u = np.random.rand()
        nu = np.random.rand()
        xi = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * nu)

        # Update position
        x = x + v * dt

        # Update velocity
        v = v - omega0**2 * x * dt - gamma * v * dt + sqrt_2kBT_gamma * xi * np.sqrt(dt)

    return x




def harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms):
    kB = 1.38e-23  # Boltzmann constant in J/K
    T = 300  # Temperature in Kelvin
    mass = vrms**2 * taup / (2 * kB * T)  # Calculating the mass from vrms and taup

    dt = t0 / steps
    x = x0
    v = v0

    gamma = 1 / taup
    sqrt_2kBT_gamma = np.sqrt(2 * kB * T * gamma)

    for _ in range(steps):
        u = np.random.rand()
        nu = np.random.rand()
        xi = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * nu)

        x = x + v * dt
        v = v - omega0**2 * x * dt - gamma * v * dt + sqrt_2kBT_gamma * xi * np.sqrt(dt)

    return x

def calculate_msd(t0, steps, taup, omega0, vrms, Navg):
    dt = t0 / steps
    msd_sum = 0.0

    for _ in range(Navg):
        x0 = np.random.normal(0, vrms * np.sqrt(taup))  # Initial position from Maxwell distribution
        v0 = np.random.normal(0, vrms)  # Initial velocity from Maxwell distribution
        
        final_position = harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms)
        msd_sum += final_position**2

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