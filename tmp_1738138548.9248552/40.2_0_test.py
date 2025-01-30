from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def second_diff(target, u, dx):
    # Adjust the array to handle boundaries using ghost cells
    if target == 0:
        u = [u[0], u[0]] + u[1:]  # Add ghost cell at the start
    elif target == len(u) - 1:
        u = u[:-1] + [u[-1], u[-1]]  # Add ghost cell at the end

    # Compute the second order derivative using the central difference formula
    deriv = (u[target + 1] - 2 * u[target] + u[target - 1]) / (dx ** 2)
    
    return deriv



# Background: Strang splitting is a numerical method used to solve partial differential equations (PDEs) by splitting the problem into simpler sub-problems that can be solved sequentially. It is particularly useful for problems involving both diffusion and advection processes. The method involves splitting the time evolution operator into parts, typically handling diffusion and advection separately, and then combining the solutions in a specific order to achieve higher accuracy. In the context of a diffusion equation, the Strang splitting can be applied by first solving the diffusion part for half a time step, then solving the advection part for a full time step, and finally solving the diffusion part again for half a time step.


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
    n = len(u)
    
    # Create a copy of u to store the updated values
    u_new = np.copy(u)
    
    # Half time step for diffusion
    dt_half = dt / 2.0
    
    # First half-step: Diffusion
    for i in range(1, n-1):
        u_new[i] = u[i] + alpha * dt_half / (dx ** 2) * (u[i+1] - 2*u[i] + u[i-1])
    
    # Full step: Advection (assuming a simple advection model, e.g., u_t + c*u_x = 0)
    # Here, we assume c = 0 for simplicity, as the problem description does not specify advection.
    # If advection is needed, replace the following line with the appropriate advection update.
    # For now, we assume no change due to advection.
    
    # Second half-step: Diffusion
    for i in range(1, n-1):
        u_new[i] = u_new[i] + alpha * dt_half / (dx ** 2) * (u_new[i+1] - 2*u_new[i] + u_new[i-1])
    
    # Return the updated solution
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