from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def init_grid(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2):


    # Initialize the temperature grid with zeros for all time steps
    temp_grid = np.zeros((Nt, Ny, Nx), dtype=float)
    
    # Initialize the diffusivity grid with zeros
    diff_grid = np.zeros((Ny, Nx), dtype=float)
    
    # Populate the first time step of the temperature grid and the diffusivity grid using a list comprehension
    temp_grid[0] = [[T1 if x <= x_split else T2 for x in range(Nx)] for _ in range(Ny)]
    diff_grid[:] = [[alpha1 if x <= x_split else alpha2 for x in range(Nx)] for _ in range(Ny)]

    return temp_grid, diff_grid



def add_dirichlet_bc(grid, time_index, bc=np.array([])):
    if bc.size == 0:
        return grid

    # Get the dimensions of the grid
    num_rows, num_cols = grid.shape[1], grid.shape[2]

    # Apply boundary conditions, avoiding the first and last row and column
    for i, j, T in bc:
        if 0 < i < num_rows - 1 and 0 < j < num_cols - 1:
            grid[time_index, i, j] = T

    return grid



def add_neumann_bc(grid, time_index, bc=np.array([])):
    if bc.size == 0:
        return grid

    num_rows, num_cols = grid.shape[1], grid.shape[2]

    # Apply Neumann boundary conditions using a logarithmic adjustment
    for i, j, grad in bc:
        if i == 0 and 0 < j < num_cols - 1:  # Top boundary, not corners
            grid[time_index, i, j] = grid[time_index, i + 1, j] - np.log1p(abs(grad))
        elif i == num_rows - 1 and 0 < j < num_cols - 1:  # Bottom boundary, not corners
            grid[time_index, i, j] = grid[time_index, i - 1, j] + np.log1p(abs(grad))
        if j == 0 and 0 < i < num_rows - 1:  # Left boundary, not corners
            grid[time_index, i, j] = grid[time_index, i, j + 1] - np.log1p(abs(grad))
        elif j == num_cols - 1 and 0 < i < num_rows - 1:  # Right boundary, not corners
            grid[time_index, i, j] = grid[time_index, i, j - 1] + np.log1p(abs(grad))

    return grid



# Background: The 2D heat equation describes how heat diffuses through a given region over time. 
# It is governed by the partial differential equation: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²), where T is the temperature, 
# t is time, and α is the thermal diffusivity. The finite difference method approximates these derivatives 
# using discrete grid points. The central difference method is used for spatial derivatives, 
# and an explicit time-stepping method is used for the time derivative. 
# To ensure numerical stability, the time step Δt is chosen as 1/(4*max(α1, α2)). 
# Dirichlet boundary conditions specify the temperature at the boundary, 
# while Neumann boundary conditions specify the temperature gradient.


def heat_equation(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2, bc_dirichlet, bc_neumann):
    # Initialize the temperature and diffusivity grids
    temp_grid, diff_grid = init_grid(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2)
    
    # Calculate the time step for stability
    dt = 1 / (4 * max(alpha1, alpha2))
    
    # Iterate over each time step
    for t in range(1, Nt):
        # Copy the previous time step to start the update
        temp_grid[t] = temp_grid[t-1].copy()
        
        # Update the temperature grid using the finite difference method
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                # Central difference for the second spatial derivatives
                d2T_dx2 = (temp_grid[t-1, i, j+1] - 2*temp_grid[t-1, i, j] + temp_grid[t-1, i, j-1])
                d2T_dy2 = (temp_grid[t-1, i+1, j] - 2*temp_grid[t-1, i, j] + temp_grid[t-1, i-1, j])
                
                # Update temperature using the heat equation
                temp_grid[t, i, j] = temp_grid[t-1, i, j] + dt * diff_grid[i, j] * (d2T_dx2 + d2T_dy2)
        
        # Apply Dirichlet boundary conditions
        temp_grid = add_dirichlet_bc(temp_grid, t, bc_dirichlet)
        
        # Apply Neumann boundary conditions
        temp_grid = add_neumann_bc(temp_grid, t, bc_neumann)
    
    return temp_grid


try:
    targets = process_hdf5_to_tuple('45.4', 3)
    target = targets[0]
    Nt = 200
    Nx = 20
    Ny = 20
    x_split = Nx//3
    T1 = 100
    alpha1 = 20
    T2 = 100
    alpha2 = 20
    bc_dirichlet = np.array([[3, 3, 200], [3, 4, 200], [4, 3, 200], [4, 4, 200]])
    bc_neumann = np.concatenate((np.array([[0, j, 10] for j in range(0, Nx)]),np.array([[i, 0, 10] for i in range(0, Ny)]), np.array([[i, Nx-1, 0] for i in range(0, Ny)]), np.array([[Ny-1, j, 0] for j in range(0, Nx)])), axis=0)
    result = heat_equation(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2, bc_dirichlet, bc_neumann)
    assert np.allclose(result[190], target)

    target = targets[1]
    Nt = 200
    Nx = 20
    Ny = 20
    x_split = Nx//2
    T1 = 80
    alpha1 = 10
    T2 = 100
    alpha2 = 20
    bc_dirichlet = np.array([])
    bc_neumann = np.concatenate((np.array([[0, j, -10] for j in range(0, Nx)]),np.array([[i, 0, -10] for i in range(0, Ny)]), np.array([[i, Nx-1, 10] for i in range(0, Ny)]), np.array([[Ny-1, j, 10] for j in range(0, Nx)])), axis=0)
    result = heat_equation(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2, bc_dirichlet, bc_neumann)
    assert np.allclose(result[190], target)

    target = targets[2]
    Nt = 200
    Nx = 20
    Ny = 20
    x_split = Nx//3
    T1 = 100
    alpha1 = 20
    T2 = 50
    alpha2 = 10
    bc_dirichlet = np.concatenate((np.array([[0, j, 10] for j in range(0, Nx)]),np.array([[i, 0, 10] for i in range(0, Ny)]), np.array([[i, Nx-1, 20] for i in range(0, Ny)]), np.array([[Ny-1, j, 20] for j in range(0, Nx)])), axis=0)
    bc_neumann = np.array([])
    result = heat_equation(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2, bc_dirichlet, bc_neumann)
    assert np.allclose(result[190], target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e