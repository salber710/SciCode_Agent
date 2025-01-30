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


try:
    targets = process_hdf5_to_tuple('40.2', 3)
    target = targets[0]
    u = np.array([-1,-1, -1, 0,1,2,3,4,5,6])
    dt  = 0.1
    dx  = 0.01
    alpha = 0.5
    assert np.allclose(Strang_splitting(u, dt, dx, alpha), target)

    target = targets[1]
    u = np.array([0,1,2,3,4,5,6])
    dt  = 0.1
    dx  = 0.1
    alpha = 0.2
    assert np.allclose(Strang_splitting(u, dt, dx, alpha), target)

    target = targets[2]
    u = np.array([0,1,2,4,6,8,0,1,23])
    dt  = 0.01
    dx  = 0.05
    alpha = -0.2
    assert np.allclose(Strang_splitting(u, dt, dx, alpha), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e