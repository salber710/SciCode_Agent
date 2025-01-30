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
    if m <= 0:
        raise ValueError("Mass must be positive.")

    # Calculate initial acceleration
    a0 = -omega**2 * x0
    
    # Update position
    xt = x0 + v0 * dt + 0.5 * a0 * dt**2
    
    # Calculate new acceleration
    at = -omega**2 * xt
    
    # Update velocity
    vt = v0 + 0.5 * (a0 + at) * dt
    
    return [vt, xt]


# Background: The Nosé-Hoover chain (NHC) is a method used in molecular dynamics to simulate systems at constant temperature.
# It extends the Nosé-Hoover thermostat by introducing a chain of thermostats, which helps in achieving better control over
# the temperature fluctuations. The Liouville operator for the NHC involves several terms that update the velocities and
# positions of the system's particles, as well as the auxiliary variables (ξ_i) and their conjugate momenta (v_ξ_i).
# The operator is applied over a time step Δt/2 to integrate these variables. The terms in the operator account for the
# coupling between the system's velocity and the thermostat variables, as well as the forces acting on the system.
# The G_i terms are derived from the equations of motion for the thermostats and depend on the kinetic energy of the system
# and the target temperature T. The integration involves updating the velocities and positions using these terms.


def nhc_step(v0, G, V, X, dt, m, T, omega):
    '''Calculate the position and velocity of the harmonic oscillator using the Nosé-Hoover-chain Liouville operator
    Input
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list of float
        The initial force constants for the thermostats.
    V : list of float
        The initial velocities of the thermostats.
    X : list of float
        The initial positions of the thermostats.
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
    G : list of float
        The updated force constants for the thermostats.
    V : list of float
        The updated velocities of the thermostats.
    X : list of float
        The updated positions of the thermostats.
    '''
    if m <= 0:
        raise ValueError("Mass must be positive")
    if T <= 0:
        raise ValueError("Temperature must be positive")

    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    # Calculate G_1
    G_1 = (1 / G[0]) * (m * v0**2 - k_B * T)

    # Update velocity of the oscillator
    v = v0 * np.exp(-V[0] * dt / 2)

    # Update the auxiliary variables and their conjugate momenta
    for i in range(len(V)):
        X[i] += V[i] * dt / 2
        if i == 0:
            V[i] += (G_1 - V[i] * V[i+1]) * dt / 2 if i < len(V) - 1 else G_1 * dt / 2
        elif i < len(V) - 1:
            G_i = (1 / G[i]) * (G[i-1] * V[i-1]**2 - k_B * T)
            V[i] += (G_i - V[i] * V[i+1]) * dt / 2
        else:
            G_M = (1 / G[i]) * (G[i-1] * V[i-1]**2 - k_B * T)
            V[i] += G_M * dt / 2

    # Update velocity of the oscillator again
    v *= np.exp(-V[0] * dt / 2)

    return v, G, V, X



# Background: Yoshida's fourth-order method is a symplectic integrator used to achieve higher accuracy in numerical simulations
# of dynamical systems. It is particularly useful in molecular dynamics and other simulations where long-term energy conservation
# is important. The method involves a sequence of updates with specific coefficients that ensure the integration is accurate
# to fourth order in the time step. In the context of the Nosé-Hoover chain, this method can be used to more accurately
# integrate the auxiliary variables (ξ_i), their conjugate momenta (v_ξ_i), and the velocity of the oscillator (v) over
# a time step Δt/2. The coefficients for Yoshida's fourth-order method are typically chosen to minimize errors and ensure
# stability, and they involve multiple sub-steps within each time step.


def nhc_Y4(v0, G, V, X, dt, m, T, omega):
    '''Use the Yoshida's fourth-order method to give a more accurate evolution of the extra variables
    Inputs:
    v0 : float
        The initial velocity of the harmonic oscillator.
    G : list of float
        The initial force constants for the thermostats.
    V : list of float
        The initial velocities of the thermostats.
    X : list of float
        The initial positions of the thermostats.
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
    G : list of float
        The updated force constants for the thermostats.
    V : list of float
        The updated velocities of the thermostats.
    X : list of float
        The updated positions of the thermostats.
    '''
    if m <= 0:
        raise ValueError("Mass must be positive")
    if T <= 0:
        raise ValueError("Temperature must be positive")

    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    # Yoshida's fourth-order coefficients
    c1 = 0.6756035959798289
    c2 = -0.1756035959798288
    c3 = 0.1756035959798288
    c4 = 0.6756035959798289

    d1 = 1.3512071919596578
    d2 = -1.7024143839193156
    d3 = 1.3512071919596578

    # Function to perform a single NHC step
    def nhc_single_step(v, G, V, X, dt):
        # Calculate G_1
        G_1 = (1 / G[0]) * (m * v**2 - k_B * T)

        # Update velocity of the oscillator
        v *= np.exp(-V[0] * dt / 2)

        # Update the auxiliary variables and their conjugate momenta
        for i in range(len(V)):
            X[i] += V[i] * dt / 2
            if i == 0:
                V[i] += (G_1 - V[i] * V[i+1]) * dt / 2 if i < len(V) - 1 else G_1 * dt / 2
            elif i < len(V) - 1:
                G_i = (1 / G[i]) * (G[i-1] * V[i-1]**2 - k_B * T)
                V[i] += (G_i - V[i] * V[i+1]) * dt / 2
            else:
                G_M = (1 / G[i]) * (G[i-1] * V[i-1]**2 - k_B * T)
                V[i] += G_M * dt / 2

        # Update velocity of the oscillator again
        v *= np.exp(-V[0] * dt / 2)

        return v, G, V, X

    # Perform Yoshida's fourth-order integration
    v, G, V, X = nhc_single_step(v0, G, V, X, c1 * dt)
    v, G, V, X = nhc_single_step(v, G, V, X, c2 * dt)
    v, G, V, X = nhc_single_step(v, G, V, X, c3 * dt)
    v, G, V, X = nhc_single_step(v, G, V, X, c4 * dt)

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
