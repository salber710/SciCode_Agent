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



def Strang_splitting(u, dt, dx, alpha):
    n = len(u)
    u_new = np.copy(u)
    dt_half = dt / 2.0

    # First half-step diffusion using a forward Euler method with reflective boundary conditions
    u_new[0] = u[0] + alpha * dt_half / dx**2 * (u[1] - u[0])
    u_new[1:-1] = u[1:-1] + alpha * dt_half / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    u_new[-1] = u[-1] + alpha * dt_half / dx**2 * (u[-2] - u[-1])

    # Apply a simple explicit advection step with reflective boundary conditions (assuming advection velocity c=1)
    c = 1
    u_adv = np.copy(u_new)
    u_adv[0] = u_new[0] - c * dt / dx * (u_new[1] - u_new[0])
    u_adv[1:-1] = u_new[1:-1] - c * dt / dx * (u_new[2:] - u_new[:-2])
    u_adv[-1] = u_new[-1] - c * dt / dx * (u_new[-1] - u_new[-2])

    # Second half-step diffusion using a forward Euler method on the advected values with reflective boundary conditions
    u_new[0] = u_adv[0] + alpha * dt_half / dx**2 * (u_adv[1] - u_adv[0])
    u_new[1:-1] = u_adv[1:-1] + alpha * dt_half / dx**2 * (u_adv[2:] - 2*u_adv[1:-1] + u_adv[:-2])
    u_new[-1] = u_adv[-1] + alpha * dt_half / dx**2 * (u_adv[-2] - u_adv[-1])

    return u_new



# Background: The diffusion-reaction equation is a partial differential equation that models the distribution of a substance within a space over time, considering both diffusion and reaction processes. The diffusion term is typically represented by a second-order spatial derivative, while the reaction term can be a function of the concentration. In this implementation, we use a second-order central difference scheme for spatial derivatives and a first-order Strang splitting scheme to separate the diffusion and reaction processes. The time integration is performed using a first-order forward Euler method, which is simple and suitable for small time steps. The Courant-Friedrichs-Lewy (CFL) condition is a stability criterion that relates the time step size to the spatial grid size and the wave speed, ensuring numerical stability.


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
    L = 1.0  # Length of the domain
    dx = np.sqrt(alpha * dt / CFL)  # Calculate spatial step size based on CFL condition
    nx = int(L / dx) + 1  # Number of spatial points
    x = np.linspace(0, L, nx)  # Spatial grid

    # Initial condition: assume a Gaussian distribution
    u = np.exp(-100 * (x - 0.5)**2)

    # Time-stepping loop
    t = 0.0
    while t < T:
        # Apply Strang splitting for diffusion and reaction
        u = Strang_splitting(u, dt, dx, alpha)
        
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