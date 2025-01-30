import numpy as np



# Background: The velocity-Verlet algorithm is a numerical method used to integrate Newton's equations of motion. 
# It is particularly useful for systems where the forces depend only on the positions, such as harmonic oscillators. 
# The algorithm updates positions and velocities in a staggered manner, which provides better stability and accuracy 
# compared to simple Euler methods. For a harmonic oscillator, the force is given by F = -k * x, where k is the 
# spring constant. The angular frequency omega is related to the spring constant and mass by omega = sqrt(k/m). 
# The velocity-Verlet algorithm involves the following steps:
# 1. Calculate the acceleration at the current position: a = -omega^2 * x.
# 2. Update the position using the current velocity and half of the acceleration: x_new = x + v * dt + 0.5 * a * dt^2.
# 3. Calculate the new acceleration at the updated position: a_new = -omega^2 * x_new.
# 4. Update the velocity using the average of the old and new accelerations: v_new = v + 0.5 * (a + a_new) * dt.

def Verlet(v0, x0, m, dt, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the velocity-Verlet algorithm
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    x0 : float
        The initial position of the harmonic oscillator.
    m : float
    dt : float
        The integration time step.
    omega: float
    Output:
    [vt, xt] : list
        The updated velocity and position of the harmonic oscillator.
    '''
    # Calculate initial acceleration
    a0 = -omega**2 * x0
    
    # Update position
    xt = x0 + v0 * dt + 0.5 * a0 * dt**2
    
    # Calculate new acceleration
    at = -omega**2 * xt
    
    # Update velocity
    vt = v0 + 0.5 * (a0 + at) * dt
    
    return [vt, xt]

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
