from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



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

    # Compute initial acceleration based on initial position
    a0 = -omega**2 * x0
    
    # Update position based solely on initial conditions
    xt = x0 + dt * v0 + (dt**2 / 2) * a0
    
    # Update velocity based on initial acceleration
    vt = v0 + dt * a0
    
    # Compute new acceleration after position update
    a1 = -omega**2 * xt
    
    # Correct the velocity using the new acceleration
    vt += 0.5 * dt * a1

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