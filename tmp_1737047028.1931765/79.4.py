import numpy as np

# Background: The velocity-Verlet algorithm is a numerical method used to integrate Newton's equations of motion. 
# It is particularly useful for systems where the forces depend only on the positions, such as harmonic oscillators. 
# The algorithm updates positions and velocities in a time-stepped manner, providing a stable and accurate solution 
# for the motion of particles. For a harmonic oscillator, the restoring force is given by F = -m * omega^2 * x, 
# where m is the mass, omega is the angular frequency, and x is the position. The velocity-Verlet algorithm 
# involves the following steps:
# 1. Calculate the acceleration at the current position: a = F/m = -omega^2 * x.
# 2. Update the position using the current velocity and half the acceleration: x_new = x + v * dt + 0.5 * a * dt^2.
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


# Background: The Nosé-Hoover chain (NHC) is a method used to simulate the canonical ensemble in molecular dynamics.
# It introduces additional variables, called thermostats, to control the temperature of the system. The Liouville
# operator for the NHC involves integrating the equations of motion for these additional variables, which include
# the positions (ξ_i) and their conjugate momenta (v_ξ_i). The operator is applied over a time step Δt/2 to update
# the velocities and positions of the system. The forces G_i are derived from the kinetic energy and the target
# temperature, ensuring that the system maintains the desired temperature. The equations of motion for the NHC
# are integrated using a symplectic method, which preserves the Hamiltonian structure of the system.


def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Input
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list of floats
        The initial forces of the harmonic oscillator.
    V : list of floats
        The initial velocities of the particles.
    X : list of floats
        The initial positions of the particles.
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
    G : list of floats
        The updated forces of the harmonic oscillator.
    V : list of floats
        The updated velocities of the particles.
    X : list of floats
        The updated positions of the particles.
    '''
    k_B = 1.0  # Boltzmann constant, assuming units where k_B = 1
    Q = [1.0 for _ in G]  # Assuming unit mass for the thermostats for simplicity

    # Update velocity of the oscillator
    v = v0 * np.exp(-0.5 * dt * V[0])

    # Update positions of the thermostats
    for i in range(len(X)):
        X[i] += 0.5 * dt * V[i]

    # Update forces G_i
    G[0] = (m * v**2 - k_B * T) / Q[0]
    for i in range(1, len(G)):
        G[i] = (Q[i-1] * V[i-1]**2 - k_B * T) / Q[i]

    # Update velocities of the thermostats
    for i in range(len(V) - 1):
        V[i] += 0.5 * dt * (G[i] - V[i] * V[i+1])
    V[-1] += 0.5 * dt * G[-1]

    # Update velocity of the oscillator again
    v *= np.exp(-0.5 * dt * V[0])

    return v, G, V, X


# Background: Yoshida's fourth-order method is a symplectic integrator used to solve differential equations
# with higher accuracy than lower-order methods. It is particularly useful in molecular dynamics simulations
# where preserving the Hamiltonian structure of the system is crucial. The method involves a sequence of
# fractional time steps with specific coefficients that ensure fourth-order accuracy. In the context of the
# Nosé-Hoover chain, this method will be used to integrate the equations of motion for the extra variables
# (ξ_i), their conjugate momenta (v_ξ_i), and the velocity of the oscillator (v) over a time step Δt/2.


def nhc_Y4(v0, G, V, X, dt, m, T, omega):
    '''Use the Yoshida's fourth-order method to give a more accurate evolution of the extra variables
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list of floats
        The initial forces of the harmonic oscillator.
    V : list of floats
        The initial velocities of the particles.
    X : list of floats
        The initial positions of the particles.
    dt : float
        The integration time step.
    m : float
        The mass of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    omega : float
        The frequency of the harmonic oscillator.
    Output:
    v : float
        The updated velocity of the harmonic oscillator.
    G : list of floats
        The updated forces of the harmonic oscillator.
    V : list of floats
        The updated velocities of the particles.
    X : list of floats
        The updated positions of the particles.
    '''

    # Yoshida coefficients for fourth-order integration
    c1 = 0.6756035959798289
    c2 = -0.1756035959798288
    c3 = 0.1756035959798288
    c4 = 0.6756035959798289

    d1 = 1.3512071919596578
    d2 = -1.7024143839193156
    d3 = 1.3512071919596578

    k_B = 1.0  # Boltzmann constant, assuming units where k_B = 1
    Q = [1.0 for _ in G]  # Assuming unit mass for the thermostats for simplicity

    def nhc_step_partial(v, G, V, X, dt):
        # Update velocity of the oscillator
        v *= np.exp(-0.5 * dt * V[0])

        # Update positions of the thermostats
        for i in range(len(X)):
            X[i] += 0.5 * dt * V[i]

        # Update forces G_i
        G[0] = (m * v**2 - k_B * T) / Q[0]
        for i in range(1, len(G)):
            G[i] = (Q[i-1] * V[i-1]**2 - k_B * T) / Q[i]

        # Update velocities of the thermostats
        for i in range(len(V) - 1):
            V[i] += 0.5 * dt * (G[i] - V[i] * V[i+1])
        V[-1] += 0.5 * dt * G[-1]

        # Update velocity of the oscillator again
        v *= np.exp(-0.5 * dt * V[0])

        return v, G, V, X

    # Apply Yoshida's fourth-order method
    v, G, V, X = nhc_step_partial(v0, G, V, X, c1 * dt / 2)
    v, G, V, X = nhc_step_partial(v, G, V, X, d1 * dt / 2)
    v, G, V, X = nhc_step_partial(v, G, V, X, c2 * dt / 2)
    v, G, V, X = nhc_step_partial(v, G, V, X, d2 * dt / 2)
    v, G, V, X = nhc_step_partial(v, G, V, X, c3 * dt / 2)
    v, G, V, X = nhc_step_partial(v, G, V, X, d3 * dt / 2)
    v, G, V, X = nhc_step_partial(v, G, V, X, c4 * dt / 2)

    return v, G, V, X



# Background: The integration of the full Liouville operator for the Nosé-Hoover-chain thermostat involves
# combining the effects of the harmonic oscillator's dynamics and the thermostat's dynamics. The operator
# is split into three parts: L1, L2, and L_NHC. L1 corresponds to the force acting on the velocity, L2
# corresponds to the velocity acting on the position, and L_NHC corresponds to the Nosé-Hoover-chain dynamics.
# The evolution of the system is performed using a sequence of operations that apply these operators in a
# specific order to ensure accurate integration. The sequence is designed to maintain the symplectic nature
# of the system, which is crucial for preserving the physical properties over long simulations. The velocity-Verlet
# algorithm is used for the L1 and L2 parts, while the Nosé-Hoover-chain dynamics are integrated using a
# symplectic method, such as Yoshida's fourth-order method, to ensure stability and accuracy.

def nose_hoover_chain(x0, v0, T, M, m, omega, dt, nsteps):
    '''Integrate the full Liouville operator of the Nose-Hoover-chain thermostat and get the trajectories of the harmonic oscillator
    Inputs:
    x0 : float
        The initial position of the harmonic oscillator.
    v0 : float
        The initial velocity of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    M : int
        The number of Nose-Hoover-chains.
    m : float
        The mass of the harmonic oscillator.
    omega : float
        The frequency of the harmonic oscillator.
    dt : float
        The integration time step.
    nsteps : int
        The number of integration time steps.
    Outputs:
    x : array of shape (nsteps, 1)
        The position trajectory of the harmonic oscillator.
    v : array of shape (nsteps, 1)
        The velocity trajectory of the harmonic oscillator.
    '''


    # Initialize arrays to store the trajectory
    x = np.zeros(nsteps)
    v = np.zeros(nsteps)

    # Initial conditions
    x[0] = x0
    v[0] = v0

    # Initialize Nosé-Hoover chain variables
    V = np.zeros(M)  # Velocities of the thermostats
    X = np.zeros(M)  # Positions of the thermostats
    G = np.zeros(M)  # Forces on the thermostats

    # Boltzmann constant
    k_B = 1.0

    # Masses of the thermostats (assuming unit mass for simplicity)
    Q = np.ones(M)

    # Function to perform a single step of the Nosé-Hoover chain integration
    def nhc_step_partial(v, G, V, X, dt):
        # Update velocity of the oscillator
        v *= np.exp(-0.5 * dt * V[0])

        # Update positions of the thermostats
        for i in range(len(X)):
            X[i] += 0.5 * dt * V[i]

        # Update forces G_i
        G[0] = (m * v**2 - k_B * T) / Q[0]
        for i in range(1, len(G)):
            G[i] = (Q[i-1] * V[i-1]**2 - k_B * T) / Q[i]

        # Update velocities of the thermostats
        for i in range(len(V) - 1):
            V[i] += 0.5 * dt * (G[i] - V[i] * V[i+1])
        V[-1] += 0.5 * dt * G[-1]

        # Update velocity of the oscillator again
        v *= np.exp(-0.5 * dt * V[0])

        return v, G, V, X

    # Main integration loop
    for step in range(1, nsteps):
        # Step 1: Apply L_NHC for dt/2
        v[step-1], G, V, X = nhc_step_partial(v[step-1], G, V, X, dt/2)

        # Step 2: Apply L1 for dt/2
        a = -omega**2 * x[step-1]  # Acceleration due to harmonic force
        v_half = v[step-1] + 0.5 * a * (dt/2)

        # Step 3: Apply L2 for dt
        x[step] = x[step-1] + v_half * dt

        # Step 4: Apply L1 for dt/2
        a_new = -omega**2 * x[step]
        v[step] = v_half + 0.5 * a_new * (dt/2)

        # Step 5: Apply L_NHC for dt/2
        v[step], G, V, X = nhc_step_partial(v[step], G, V, X, dt/2)

    return x, v


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('79.4', 3)
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
nsteps = N
assert cmp_tuple_or_list(nose_hoover_chain(x0, v0, T0, M, m, omega, dt, nsteps), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
T0 = 0.1
v0 = np.sqrt(2 * T0) * 2
x0 = 0.0
N = 20000
M = 2
m = 1
omega = 1
dt = 0.1
nsteps = N
assert cmp_tuple_or_list(nose_hoover_chain(x0, v0, T0, M, m, omega, dt, nsteps), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
T0 = 0.1
v0 = np.sqrt(2 * T0) * 2
x0 = 0.0
N = 40000
M = 2
m = 1
omega = 1
dt = 0.1
nsteps = N
assert cmp_tuple_or_list(nose_hoover_chain(x0, v0, T0, M, m, omega, dt, nsteps), target)
