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
    
    # Number of spatial steps
    n = len(u)

    # Copy the current solution to apply updates
    u_star = np.copy(u)
    
    # First half-step for the diffusion term
    for i in range(1, n-1):
        u_star[i] = u[i] + 0.5 * alpha * dt / (dx ** 2) * (u[i+1] - 2 * u[i] + u[i-1])
    
    # Handle boundary conditions (Dirichlet)
    u_star[0] = u[0]  # Assuming Dirichlet boundary condition at the left boundary
    u_star[-1] = u[-1]  # Assuming Dirichlet boundary condition at the right boundary

    # Reaction term with a full step (example: linear reaction term f(u) = -bu)
    b = 1  # Example reaction rate constant
    u_star = u_star - dt * b * u_star

    # Second half-step for the diffusion term
    u_new = np.copy(u_star)
    for i in range(1, n-1):
        u_new[i] = u_star[i] + 0.5 * alpha * dt / (dx ** 2) * (u_star[i+1] - 2 * u_star[i] + u_star[i-1])
    
    # Handle boundary conditions (Dirichlet)
    u_new[0] = u_star[0]  # Assuming Dirichlet boundary condition at the left boundary
    u_new[-1] = u_star[-1]  # Assuming Dirichlet boundary condition at the right boundary

    return u_new


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