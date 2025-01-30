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
    
    # Number of spatial points
    N = len(u)
    
    # 1st half step: solve diffusion equation with theta method
    u_star = np.zeros_like(u)
    r = alpha * dt / (2 * dx**2)
    
    A = np.eye(N) * (1 + 2 * r) + np.eye(N, k=-1) * (-r) + np.eye(N, k=1) * (-r)
    u_star[:] = np.linalg.solve(A, u)
    
    # 2nd half step: solve reaction equation with forward Euler
    # Assuming linear reaction term f(u) = -bu for simplicity
    b = 1.0  # Reaction rate constant
    
    u_star = u_star - dt * 0.5 * b * u_star
    
    # 3rd half step: solve diffusion equation again with theta method
    u_check = np.zeros_like(u_star)
    u_check[:] = np.linalg.solve(A, u_star)
    
    # 4th half step: solve reaction equation again with forward Euler
    u_check = u_check - dt * 0.5 * b * u_check
    
    return u_check


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