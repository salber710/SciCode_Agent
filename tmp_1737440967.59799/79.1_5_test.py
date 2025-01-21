import numpy as np



# Background: The velocity-Verlet algorithm is a numerical method used to integrate Newton's equations of motion. 
# It is particularly well-suited for simulations in classical mechanics, such as the motion of a harmonic oscillator.
# In the context of a harmonic oscillator with mass m and angular frequency omega, the restoring force is given by 
# F = -m * omega^2 * x, which follows Hooke's Law. 
# The velocity-Verlet algorithm updates the position and velocity using the following steps:
# 1. Calculate the new position using the current velocity and acceleration: 
#    x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
# 2. Calculate the acceleration at the new position: 
#    a(t + dt) = -omega^2 * x(t + dt)
# 3. Calculate the new velocity using the average of the current and new acceleration:
#    v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt

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
    
    # Calculate new acceleration at xt
    a1 = -omega**2 * xt
    
    # Update velocity
    vt = v0 + 0.5 * (a0 + a1) * dt
    
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
