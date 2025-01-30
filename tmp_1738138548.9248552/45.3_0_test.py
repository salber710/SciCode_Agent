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



# Background: Neumann boundary conditions specify the derivative (gradient) of a function on a boundary, rather than the function's value itself. 
# In the context of a heat equation, this means specifying the heat flux across the boundary. 
# For a discrete grid, the gradient can be approximated using finite differences. 
# Depending on the boundary, we use either a forward difference (for the left and top boundaries) or a backward difference (for the right and bottom boundaries).
# The Neumann condition is applied by adjusting the temperature at the boundary based on the specified gradient.


def add_neumann_bc(grid, time_index, bc=np.array([])):
    '''Add Neumann type of boundary conditions to the temperature grid. Users define the real space positions and values of the boundary conditions.
    This function will update boundary conditions constantly as time progresses.
    Boundary conditions will not be applied to corner grid points.
    Input
    grid: the temperature grid for the problem; 3d array of floats
    time_index: the function will update the boundary conditions for the slice with this time axis index; int
    bc: a 2d array where each row has three elements: i, j, and T. 
        - i and j: the row and column indices to set the boundary conditions; ints
        - T: the value of the boundary condition; float
    Output
    grid: the updated temperature grid; 3d array of floats
    '''
    if bc.size == 0:
        return grid

    # Get the dimensions of the grid
    num_rows, num_cols = grid.shape[1], grid.shape[2]

    # Apply Neumann boundary conditions
    for i, j, grad in bc:
        if 0 < i < num_rows - 1 and 0 < j < num_cols - 1:
            # Check if the boundary is on the left or right
            if i == 0:  # Top boundary
                grid[time_index, i, j] = grid[time_index, i + 1, j] - grad
            elif i == num_rows - 1:  # Bottom boundary
                grid[time_index, i, j] = grid[time_index, i - 1, j] + grad
            # Check if the boundary is on the top or bottom
            if j == 0:  # Left boundary
                grid[time_index, i, j] = grid[time_index, i, j + 1] - grad
            elif j == num_cols - 1:  # Right boundary
                grid[time_index, i, j] = grid[time_index, i, j - 1] + grad

    return grid


try:
    targets = process_hdf5_to_tuple('45.3', 3)
    target = targets[0]
    assert np.allclose(add_neumann_bc(init_grid(2, 10, 5, 2, 100, 1, 50, 2)[0], 0, [[1, 0, 20]]), target)

    target = targets[1]
    assert np.allclose(add_neumann_bc(init_grid(2, 10, 5, 2, 100, 1, 50, 2)[0], 0, np.array([[0, j, 40] for j in range(0, 10)])), target)

    target = targets[2]
    assert np.allclose(add_neumann_bc(init_grid(2, 10, 5, 2, 10, 1, 20, 2)[0], 0, np.array([[i, 0, 100] for i in range(0, 5)])), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e