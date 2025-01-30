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

    # Calculate indices for a centered difference with conditional ghost cell logic
    left_idx = target - 1 if target > 0 else 0
    right_idx = target + 1 if target < n - 1 else n - 1

    # Use centered difference formula for second derivative
    deriv = (u[right_idx] - 2 * u[target] + u[left_idx]) / (dx * dx)

    return deriv



def Strang_splitting(u, dt, dx, alpha):
    '''Inputs:
    u : solution, array of float
    dt: time interval, float
    dx: spatial interval, float
    alpha: diffusive coefficient, float
    Outputs:
    u : solution, array of float
    '''

    # Function to apply absorbing boundary conditions
    def apply_absorbing_boundary_conditions(u):
        u[0] = 0
        u[-1] = 0
        return u

    # Function to perform a diffusion step using a leapfrog method
    def leapfrog_diffusion_step(u, u_prev, dt, dx, alpha):
        n = len(u)
        u_new = np.copy(u)
        coeff = alpha * dt / (dx**2)
        
        # Leapfrog update for inner points
        for i in range(1, n - 1):
            u_new[i] = u_prev[i] + 2 * coeff * (u[i + 1] - 2 * u[i] + u[i - 1])
        
        return apply_absorbing_boundary_conditions(u_new)
    
    # Initialize the previous solution for the leapfrog method
    u_prev = np.copy(u)

    # First half-step diffusion
    u_half = leapfrog_diffusion_step(u, u_prev, dt / 2, dx, alpha)
    
    # Placeholder for full step of another process (identity for now)
    u_full = np.copy(u_half)  # Assuming no change for the other process

    # Update u_prev for second half-step
    u_prev = np.copy(u_half)
    
    # Second half-step diffusion
    u_final = leapfrog_diffusion_step(u_full, u_prev, dt / 2, dx, alpha)
    
    return u_final




def solve(CFL, T, dt, alpha):
    '''Inputs:
    CFL : Courant-Friedrichs-Lewy condition number
    T   : Max time, float
    dt  : Time interval, float
    alpha : diffusive coefficient , float
    Outputs:
    u   : solution, array of float
    '''
    
    # Define spatial domain
    L = 1.0  # Length of the domain
    nx = max(int(L / (CFL * dt / alpha)), 3)  # Ensure at least 3 points for spatial derivative
    dx = L / (nx - 1)  # Spatial interval

    # Initialize solution array
    u = np.zeros(nx)
    
    # Initial condition: Linear gradient
    u = np.linspace(0, 1, nx)

    # Function to calculate the second order spatial derivative using convolution
    def second_order_diffusion(u, dx):
        kernel = np.array([1, -2, 1]) / (dx * dx)
        return np.convolve(u, kernel, mode='same')

    # Function to apply Neumann boundary conditions (zero gradient at boundaries)
    def apply_neumann_boundary_conditions(u):
        u[0] = u[1]
        u[-1] = u[-2]
        return u

    # Time-stepping loop
    t = 0.0
    while t < T:
        # Compute diffusion term
        dudx2 = second_order_diffusion(u, dx)
        
        # First-order Strang splitting - First half diffusion step
        u_half = u + 0.5 * dt * alpha * dudx2
        
        # Apply boundary conditions
        u_half = apply_neumann_boundary_conditions(u_half)

        # Placeholder for reaction term - Assuming a simple logistic reaction R(u) = u * (1 - u)
        u_reacted = u_half + dt * u_half * (1 - u_half)

        # Second half diffusion step
        dudx2_half = second_order_diffusion(u_reacted, dx)
        u = u_reacted + 0.5 * dt * alpha * dudx2_half
        
        # Apply boundary conditions
        u = apply_neumann_boundary_conditions(u)

        # Update time
        t += dt

    return u


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