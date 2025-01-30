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

    # Define constants
    gamma = 1 / taup
    dt = t0 / steps  # Time step
    sqrt_dt = np.sqrt(dt)
    
    # Initialize position and velocity
    x = x0
    v = v0
    
    # Iterate over each time step
    for _ in range(steps):
        # Generate random numbers for Gaussian process
        u = np.random.uniform(0, 1)
        nu = np.random.uniform(0, 1)
        gaussian_noise = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * nu)
        
        # Update velocity using the leapfrog method
        v += (-omega0**2 * x - gamma * v + vrms * gaussian_noise) * dt
        
        # Update position
        x += v * dt
    
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