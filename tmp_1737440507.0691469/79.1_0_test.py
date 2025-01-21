import numpy as np



# Background: The velocity-Verlet algorithm is a numerical integration method used to simulate the motion of particles.
# It is particularly suitable for systems where the force depends only on position, such as in a harmonic oscillator.
# The basic idea is to update both velocity and position of a particle in a staggered way, providing good energy conservation.
# The equations used in the velocity-Verlet algorithm for a harmonic oscillator are:
# 1. Compute the acceleration a(t) = -omega^2 * x(t) (harmonic restoring force divided by mass is a = F/m = -k*x/m = -omega^2*x)
# 2. Update position: x(t + dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt^2
# 3. Compute the new acceleration a(t + dt) = -omega^2 * x(t + dt)
# 4. Update velocity: v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt

def Verlet(v0, x0, m, dt, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the velocity-Verlet algorithm
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    x0 : float
        The initial position of the harmonic oscillator.
    m : float
        The mass of the harmonic oscillator.
    dt : float
        The integration time step.
    omega: float
        The angular frequency of the harmonic oscillator.
    Output:
    [vt, xt] : list
        The updated velocity and position of the harmonic oscillator.
    '''
    
    # Initial acceleration
    a0 = -omega**2 * x0
    
    # Update position
    xt = x0 + v0 * dt + 0.5 * a0 * dt**2
    
    # New acceleration at the new position
    at = -omega**2 * xt
    
    # Update velocity
    vt = v
from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('79.1', 3)
target = targets[0]

v0 = np.sqrt(2 * 0.1)
x0 = 0.0
m = 1.0
omega = 1.0
dt = 0.1
assert np.allclose(Verlet(v0, x0, m, dt, omega), target)
target = targets[1]

v0 = np.sqrt(2 * 0.1)
x0 = 0.0
m = 1.0
omega = 1.0
dt = 0.01
assert np.allclose(Verlet(v0, x0, m, dt, omega), target)
target = targets[2]

v0 = np.sqrt(2)
x0 = 0.0
m = 1.0
omega = 1.0
dt = 0.001
assert np.allclose(Verlet(v0, x0, m, dt, omega), target)
