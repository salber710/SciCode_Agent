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
    
    # Calculate initial acceleration
    a0 = -omega**2 * x0
    
    # Update velocity to a midpoint value
    v_mid = v0 + 0.5 * a0 * dt
    
    # Update position using the midpoint velocity
    xt = x0 + v_mid * dt
    
    # Calculate new acceleration at the updated position
    a1 = -omega**2 * xt
    
    # Update velocity to the full step using the new acceleration
    vt = v_mid + 0.5 * a1 * dt
    
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