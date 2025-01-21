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


# Background: The Nosé-Hoover chain (NHC) method is a deterministic approach used in molecular dynamics to maintain a constant temperature (isothermal conditions) by coupling the system to a series of thermostats. 
# It involves integrating extra variables (thermostat variables) and their conjugate momenta alongside the system's physical variables.
# The Liouville operator for the NHC method involves updating these variables in a staggered manner to simulate the effect of the thermostats.
# Key components of the Liouville operator include:
# - Updating the system's velocity `v` using the first thermostat variable `v_{\xi_1}`.
# - Updating each thermostat variable `\xi_i` using its conjugate momentum `v_{\xi_i}`.
# - Calculating forces `G_i` based on the kinetic energy and target temperature, which then influence changes in `v_{\xi_i}`.


def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Input
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : np.ndarray
        The forces of the harmonic oscillator (array of size M, where M is the number of thermostats).
    V : np.ndarray
        The velocities of the thermostat variables (array of size M+1).
    X : np.ndarray
        The positions of the thermostat variables (array of size M).
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
    G : np.ndarray
        The updated forces of the harmonic oscillator.
    V : np.ndarray
        The updated velocities of the thermostat variables.
    X : np.ndarray
        The updated positions of the thermostat variables.
    '''
    
    # Constants
    k_B = 1.0  # Boltzmann constant (assuming units where k_B = 1 for simplicity)
    
    # Number of thermostats
    M = len(X)
    
    # Update velocity of the oscillator using the first thermostat variable
    v = v0 * np.exp(-V[0] * dt / 2.0)
    
    # Update the positions of the thermostat variables
    for i in range(M):
        X[i] += V[i] * dt / 2.0
    
    # Calculate forces G_i
    G[0] = (m * v**2 - k_B * T) / X[0]
    for i in range(1, M):
        G[i] = (X[i-1] * V[i-1]**2 - k_B * T) / X[i]
    
    # Update velocities of the thermostat variables
    for i in range(M):
        V[i] += G[i] * dt / 2.0
    
    # Update velocity of the oscillator again using the first thermostat variable
    v *= np.exp(-V[0] * dt / 2.0)
    
    # Update positions of the thermostat variables again
    for i in range(M):
        X[i] += V[i] * dt / 2.0
    
    # Update velocities of the thermostat variables again
    for i in range(M):
        V[i] += G[i] * dt / 2.0
    
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
