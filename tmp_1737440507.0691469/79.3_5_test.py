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

# Background: The Nosé-Hoover-chain (NHC) Liouville operator is used to integrate a system of particles maintained at a constant temperature. 
# This method is useful for simulating the canonical ensemble (constant NVT). The NHC method extends the Nosé-Hoover thermostat by 
# introducing a chain of thermostats, each with its own 'mass' (Q_k) and conjugate momentum (v_ξ_k), to better control temperature fluctuations.
# The operator acts by updating the velocities and momenta of the system particles and the additional variables ξ_i and v_ξ_i, which represent
# the thermostat variables. The equations involve the calculation of G_i terms, which represent the 'forces' acting due to the thermostats.
# G_1 is related to the physical system's kinetic energy and the target temperature, while the subsequent G_k terms depend on the thermostat variables themselves.


def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Input
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list of floats
        The initial forces of the harmonic oscillator related to the thermostats.
    V : list of floats
        The initial velocities of the particles and thermostat variables.
    X : list of floats
        The initial positions of the particles and thermostat variables.
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
        The updated velocities of the particles and thermostat variables.
    X : list of floats
        The updated positions of the particles and thermostat variables.
    '''

    # Constants
    k_B = 1.0  # Boltzmann constant, assume units where k_B = 1
    Q = [1.0 for _ in range(len(G))]  # Masses for the thermostats, assume all Q_k = 1 for simplicity

    # Update velocity of the oscillator
    v = v0 + (-V[0] * v0) * (dt / 2)

    # Update thermostat positions
    for i in range(len(X)):
        X[i] += V[i] * (dt / 2)

    # Update forces G
    G[0] = (m * v**2 - k_B * T) / Q[0]
    for k in range(1, len(G)):
        G[k] = (Q[k-1] * V[k-1]**2 - k_B * T) / Q[k]

    # Update thermostat velocities
    for i in range(len(V) - 1):
        V[i] += (G[i] - V[i] * V[i+1]) * (dt / 2)
    V[-1] += G[-1] * (dt / 2)

    # Update velocity of the oscillator again
    v += (-V[0] * v) * (dt / 2)

    return v, G, V, X



# Background: Yoshida's fourth-order method is a symplectic integrator used to solve differential equations with high accuracy.
# It is particularly effective for Hamiltonian systems, such as those encountered in molecular dynamics, where energy conservation
# and long-term stability are important. This method improves upon the basic integration methods by reducing the error
# associated with the time evolution of the system. The fourth-order method is achieved by taking a sequence of staggered
# steps, each weighted to balance the error terms, resulting in a more accurate simulation over larger time steps.
# In the context of the Nosé-Hoover-chain Liouville operator, this method will improve the accuracy of the integration
# of the thermostat variables and the velocity of the system.


def nhc_Y4(v0, G, V, X, dt, m, T, omega):
    '''Use the Yoshida's fourth-order method to give a more accurate evolution of the extra variables
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list of floats
        The initial forces related to the thermostats.
    V : list of floats
        The initial velocities of the particles and thermostat variables.
    X : list of floats
        The initial positions of the particles and thermostat variables.
    dt : float
        The integration time step.
    m : float
        The mass of the harmonic oscillator.
    T : float
        The temperature of the system.
    omega : float
        The frequency of the harmonic oscillator.
    Output:
    v : float
        The updated velocity of the harmonic oscillator.
    G : list of floats
        The updated forces.
    V : list of floats
        The updated velocities.
    X : list of floats
        The updated positions.
    '''
    # Yoshida coefficients for fourth-order integrator
    w0 = 1.0 / (2 - 2**(1/3))
    w1 = -2**(1/3) * w0
    c = [w0, w1, w1, w0]
    d = [0.5 * w0, 0.5 * (w0 + w1), 0.5 * (w1 + w0), 0.5 * w0]

    # Constants
    k_B = 1.0  # Boltzmann constant, assume units where k_B = 1
    Q = [1.0 for _ in range(len(G))]  # Masses for the thermostats, assume all Q_k = 1 for simplicity

    # Initialize the velocity of the oscillator
    v = v0

    # Perform the Yoshida method steps
    for i in range(4):
        # Update velocity of the oscillator
        v += (-V[0] * v) * (d[i] * dt)

        # Update thermostat positions
        for j in range(len(X)):
            X[j] += V[j] * (c[i] * dt)

        # Update forces G
        G[0] = (m * v**2 - k_B * T) / Q[0]
        for k in range(1, len(G)):
            G[k] = (Q[k-1] * V[k-1]**2 - k_B * T) / Q[k]

        # Update thermostat velocities
        for j in range(len(V) - 1):
            V[j] += (G[j] - V[j] * V[j+1]) * (c[i] * dt)
        V[-1] += G[-1] * (c[i] * dt)

    # Update velocity of the oscillator for the final half step
    v += (-V[0] * v) * (d[-1] * dt)

    return v, G, V, X

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('79.3', 3)
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
assert cmp_tuple_or_list(nhc_Y4(v0, G, V, X, dt, m, T, omega), target)
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
assert cmp_tuple_or_list(nhc_Y4(v0, G, V, X, dt, m, T, omega), target)
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
assert cmp_tuple_or_list(nhc_Y4(v0, G, V, X, dt, m, T, omega), target)
