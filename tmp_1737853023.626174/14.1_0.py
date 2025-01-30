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

from scicode.parse.parse import process_hdf5_to_tuple
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
