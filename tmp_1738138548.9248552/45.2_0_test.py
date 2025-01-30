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



# Background: Dirichlet boundary conditions are a type of boundary condition used in differential equations where the value of the function is specified on the boundary of the domain. In the context of a heat equation, this means setting the temperature at specific grid points to a fixed value. These conditions are crucial for simulating realistic scenarios where the temperature at the boundary is controlled or known. When applying these conditions to a grid, care must be taken not to apply them to corner points if specified, as these might be handled differently or left free to evolve according to the simulation.


def add_dirichlet_bc(grid, time_index, bc=np.array([])):
    '''Add Dirichlet type of boundary conditions to the temperature grid. Users define the real space positions and values of the boundary conditions. 
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
    # Iterate over each boundary condition provided
    for condition in bc:
        i, j, T = condition
        # Check if the point is not a corner point
        if not ((i == 0 or i == grid.shape[1] - 1) and (j == 0 or j == grid.shape[2] - 1)):
            # Apply the boundary condition to the specified grid point at the given time index
            grid[time_index, i, j] = T

    return grid


try:
    targets = process_hdf5_to_tuple('45.2', 3)
    target = targets[0]
    assert np.allclose(add_dirichlet_bc(init_grid(2, 10, 5, 2, 100, 1, 50, 2)[0], 0, [[1, 0, 20]]), target)

    target = targets[1]
    assert np.allclose(add_dirichlet_bc(init_grid(2, 10, 5, 2, 100, 1, 50, 2)[0], 0, np.array([[0, j, 40] for j in range(0, 10)])), target)

    target = targets[2]
    assert np.allclose(add_dirichlet_bc(init_grid(2, 10, 5, 2, 10, 1, 20, 2)[0], 0, np.array([[i, 0, 100] for i in range(0, 5)])), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e