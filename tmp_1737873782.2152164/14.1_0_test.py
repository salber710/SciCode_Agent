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


try:
    targets = process_hdf5_to_tuple('14.1', 3)
    target = targets[0]
    x0 = 0
    v0 = 0
    t0 = 1e-4
    steps = 200000
    taup = 48.5e-6
    omega0 = 2 * np.pi * 3064
    vrms = 1.422e-2
    np.random.seed(0)
    assert np.allclose(harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms), target)

    target = targets[1]
    x0 = 0
    v0 = 1.422e-2
    t0 = 2e-4
    steps = 200000
    taup = 48.5e-6
    omega0 = 2 * np.pi * 3064
    vrms = 1.422e-2
    np.random.seed(1)
    assert np.allclose(harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms), target)

    target = targets[2]
    x0 = 0
    v0 = 0
    t0 = 4e-4
    steps = 200000
    taup = 147.3e-6
    omega0 = 2 * np.pi * 3168
    vrms = 1.422e-2
    np.random.seed(1)
    assert np.allclose(harmonic_mannella_leapfrog(x0, v0, t0, steps, taup, omega0, vrms), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e