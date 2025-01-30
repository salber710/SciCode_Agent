from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The velocity-Verlet algorithm is an integration method used to solve the equations of motion in molecular dynamics simulations.
# It is particularly well-suited for systems with conservative forces, such as those governed by harmonic potentials.
# In the context of a harmonic oscillator, the force is given by Hooke's Law as F = -kx, where k is the spring constant.
# This force can also be expressed in terms of angular frequency (omega) as F = -m * omega^2 * x, where m is the mass of the oscillator.
# The velocity-Verlet algorithm updates the position and velocity of the oscillator over a small time step (dt) by first updating the position,
# then computing the new acceleration, and finally updating the velocity. The steps are:
# 1. Compute the acceleration at the current position: a(t) = -omega^2 * x(t)
# 2. Update the position using the current velocity and half the acceleration: x(t + dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt^2
# 3. Compute the new acceleration at the updated position: a(t + dt) = -omega^2 * x(t + dt)
# 4. Update the velocity using the average of the current and new acceleration: v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt

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

    # Compute initial acceleration
    a0 = -omega**2 * x0
    
    # Update position
    xt = x0 + v0 * dt + 0.5 * a0 * dt**2
    
    # Compute new acceleration at updated position
    a1 = -omega**2 * xt
    
    # Update velocity
    vt = v0 + 0.5 * (a0 + a1) * dt
    
    return [vt, xt]


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e