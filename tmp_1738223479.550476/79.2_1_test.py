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

    # Calculate initial acceleration using the harmonic force equation
    a0 = -omega**2 * x0
    
    # Use the initial acceleration to calculate a predicted velocity at the midpoint
    v_mid = v0 + a0 * dt / 2
    
    # Update the position using this midpoint velocity
    xt = x0 + v_mid * dt
    
    # Compute new acceleration at the updated position for the correction
    a1 = -omega**2 * xt
    
    # Correct the velocity using the new acceleration
    vt = v_mid + a1 * dt / 2
    
    return [vt, xt]



def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Input
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list
        The initial force terms of the harmonic oscillator.
    V : list
        The initial velocities of the particles (thermostat variables).
    X : list
        The initial positions of the particles (thermostat variables).
    dt : float
        The integration time step.
    m : float
        The mass of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    omega : float
        The frequency of the harmonic oscillator.
    Output
    v : float
        The updated velocity of the harmonic oscillator.
    G : list
        The updated force terms of the harmonic oscillator.
    V : list
        The updated velocities of the particles.
    X : list
        The updated positions of the particles.
    '''
    
    # Constants
    kB = 1.0  # Boltzmann constant
    Q = [1.0 for _ in X]  # Example mass-like parameters for thermostat variables

    # Helper function to update a single thermostat
    def update_thermostat(v_xi, xi, g, dt):
        v_xi_new = v_xi + g * dt / 2
        xi_new = xi + v_xi_new * dt / 2
        return v_xi_new, xi_new

    # Update velocity and position of the harmonic oscillator
    v = v0
    for i in range(len(V)):
        # Calculate G terms
        if i == 0:
            G[i] = (m * v**2 - kB * T) / Q[i]
        else:
            G[i] = (Q[i-1] * V[i-1]**2 - kB * T) / Q[i]

        # Update thermostat variables
        V[i], X[i] = update_thermostat(V[i], X[i], G[i], dt)

    # Apply Nosé-Hoover chain operator on the oscillator velocity
    v *= (1 - V[0] * dt / 2)

    return v, G, V, X


try:
    targets = process_hdf5_to_tuple('79.2', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    T0 = 0.1
    v0 = np.sqrt(2 * T0) * 2
    x0 = 0.0
    N = 20000
    M = 1
    m = 1
    omega = 1
    dt = 0.1
    G = np.zeros(M)
    V = np.zeros(M)
    X = np.zeros(M)
    T = T0
    assert cmp_tuple_or_list(nhc_step(v0, G, V, X, dt, m, T, omega), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    T0 = 0.1
    v0 = np.sqrt(2 * T0) * 2
    x0 = 0.0
    M = 1
    m = 1
    omega = 1
    dt = 0.01
    G = np.zeros(M)
    V = np.zeros(M)
    X = np.zeros(M)
    T = T0
    assert cmp_tuple_or_list(nhc_step(v0, G, V, X, dt, m, T, omega), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    T0 = 0.1
    v0 = np.sqrt(2 * T0) * 2
    x0 = 0.0
    M = 2
    m = 1
    omega = 1
    dt = 0.1
    G = np.zeros(M)
    V = np.zeros(M)
    X = np.zeros(M)
    T = T0
    assert cmp_tuple_or_list(nhc_step(v0, G, V, X, dt, m, T, omega), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e