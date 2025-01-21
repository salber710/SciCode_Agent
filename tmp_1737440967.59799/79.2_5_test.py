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


# Background: The Nosé-Hoover chain (NHC) method is used to control the temperature in molecular dynamics simulations.
# It introduces additional dynamical variables (ξ and v_ξ) which act as thermostats, allowing the system to simulate
# a canonical ensemble (constant temperature). The Liouville operator for the NHC includes terms that update these
# variables and the velocities of the system to ensure energy exchange consistent with a target temperature. 
# The update for half a time step is crucial for splitting the dynamics due to the staggered integration of velocities
# and positions. The forces, G_i, in the chains are derived from the kinetic energy (related to the temperature) and 
# are used to evolve the conjugate momenta, v_ξ.

def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Input
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : np.array
        The current forces G_i in the Nosé-Hoover chain.
    V : np.array
        The current velocities v_ξ_i in the Nosé-Hoover chain.
    X : np.array
        The current positions ξ_i in the Nosé-Hoover chain.
    dt : float
        The integration time step.
    m : float
        The mass of the harmonic oscillator.
    T : float
        The temperature of the harmonic oscillator.
    omega : float
        The angular frequency of the harmonic oscillator.
    Output
    v : float
        The updated velocity of the harmonic oscillator.
    G : np.array
        The updated forces of the Nosé-Hoover chain.
    V : np.array
        The updated velocities of the Nosé-Hoover chain.
    X : np.array
        The updated positions of the Nosé-Hoover chain.
    '''

    # Constants
    k_B = 1.0  # Boltzmann constant, can be set to 1 for reduced units

    # Update the velocity of the harmonic oscillator
    v = v0 * np.exp(-V[0] * dt / 2)

    # Update the positions ξ_i
    X = X + V * (dt / 2)

    # Calculate new G values
    G[0] = (m * v**2 - k_B * T) / m
    for i in range(1, len(G)):
        G[i] = (X[i-1] * V[i-1]**2 - k_B * T) / m

    # Update the velocities v_ξ_i
    for i in range(len(V) - 1):
        V[i] = V[i] + (G[i] - V[i] * V[i+1]) * (dt / 2)
    V[-1] = V[-1] + G[-1] * (dt / 2)

    # Update the velocity of the harmonic oscillator again
    v = v * np.exp(-V[0] * dt / 2)

    return v, G, V, X

from scicode.parse.parse import process_hdf5_to_tuple
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
