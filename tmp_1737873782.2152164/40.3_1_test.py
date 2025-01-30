from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def second_diff(target, u, dx):
    '''Inputs:
    target : Target cell index, int
    u      : Approximated solution value, array of floats with minimum length of 5
    dx     : Spatial interval, float
    Outputs:
    deriv  : Second order derivative of the given target cell, float
    '''

    # Length of the input array
    n = len(u)

    # If the target is at the boundary, use the ghost cell approach
    if target == 0:
        # Left boundary
        left_ghost_value = u[0]
        deriv = (u[target + 1] - 2 * u[target] + left_ghost_value) / (dx ** 2)
    elif target == n - 1:
        # Right boundary
        right_ghost_value = u[n - 1]
        deriv = (right_ghost_value - 2 * u[target] + u[target - 1]) / (dx ** 2)
    else:
        # Interior points, use central difference
        deriv = (u[target + 1] - 2 * u[target] + u[target - 1]) / (dx ** 2)

    return deriv



def Strang_splitting(u, dt, dx, alpha):
    '''Inputs:
    u : solution, array of float
    dt: time interval , float
    dx: spatial interval, float
    alpha: diffusive coefficient, float
    Outputs:
    u : solution, array of float
    '''

    # Half time step for Strang splitting
    dt_half = dt / 2.0

    # Number of spatial points
    N = len(u)

    # Create a temporary array for intermediate results
    u_star = np.zeros_like(u)
    
    # First half-step: solve the diffusion equation over dt/2
    for i in range(1, N-1):
        u_star[i] = u[i] + alpha * dt_half / (dx * dx) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Enforce boundary conditions (assume Dirichlet boundaries)
    u_star[0] = u[0]
    u_star[-1] = u[-1]
    
    # Full step: solve the reaction equation over dt
    # Assuming a linear reaction term f(u) = -b*u for demonstration
    b = 1.0  # reaction rate, this should be defined in the larger context
    u_double_star = np.zeros_like(u)
    for i in range(N):
        u_double_star[i] = u_star[i] - b * dt * u_star[i]

    # Second half-step: solve the diffusion equation over dt/2
    for i in range(1, N-1):
        u[i] = u_double_star[i] + alpha * dt_half / (dx * dx) * (u_double_star[i+1] - 2*u_double_star[i] + u_double_star[i-1])
    
    # Enforce boundary conditions (assume Dirichlet boundaries)
    u[0] = u_double_star[0]
    u[-1] = u_double_star[-1]

    return u




def solve(CFL, T, dt, alpha):
    '''Inputs:
    CFL : Courant-Friedrichs-Lewy condition number
    T   : Max time, float
    dt  : Time interval, float
    alpha : diffusive coefficient , float
    Outputs:
    u   : solution, array of float
    '''
    # Define the spatial domain
    L = 10  # Length of the domain
    dx = np.sqrt(alpha * dt / CFL)  # Compute spatial interval from CFL condition
    N = int(L / dx) + 1  # Number of spatial points
    x = np.linspace(0, L, N)  # Spatial grid

    # Initialize the solution array
    u = np.zeros(N)
    u_1 = np.zeros(N)

    # Initial condition: a stationary front
    u[x <= 0] = -1.0
    u[x > 0] = 1.0

    # Time-stepping loop
    num_steps = int(T / dt)
    for step_no in range(num_steps):
        # Perform one step of Strang splitting
        u = Strang_splitting(u, dt, dx, alpha)

        # Swap u and u_1 for the next iteration
        u_1, u = u, u_1

    return u_1


try:
    targets = process_hdf5_to_tuple('40.3', 3)
    target = targets[0]
    CFL = 0.2
    T   = 0.1
    dt  = 0.01
    alpha = 0.1
    assert np.allclose(solve(CFL, T, dt, alpha), target)

    target = targets[1]
    CFL = 0.3
    T   = 0.3
    dt  = 0.05
    alpha = 0.05
    assert np.allclose(solve(CFL, T, dt, alpha), target)

    target = targets[2]
    CFL = 0.1
    T   = 0.5
    dt  = 0.01
    alpha = 0.2
    assert np.allclose(solve(CFL, T, dt, alpha), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e