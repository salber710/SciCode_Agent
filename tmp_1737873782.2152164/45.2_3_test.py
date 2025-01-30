from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def init_grid(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2):
    '''Initialize a 3d array for storing the temperature values. The array has one time dimension and two real space dimensions.
    Initialize a 2d array for storing the thermal diffusivity corresponding to each grid point in real space.  
    There could be a vertical boundary in real space that represents the interface between two different materials.
    Input
    Nt: time dimension of the temperature grid; int
    Nx: x-dimension (number of columns) of the temperature grid; int
    Ny: y-dimension (number of rows) of the temperature grid; int
    x_split: the column index of the vertical interface. All columns up to and including this index (material 1) will have T1 and alpha1, and all columns with a larger index (material 2) will have T2 and alpha2; int
    T1: the initial temperature of each grid point for material 1(in Celsius); float
    alpha1: the thermal diffusivity of material 1; float
    T2: the initial temperature of each grid point for material 2(in Celsius); float
    alpha2: the heat conductivity of material 2; float
    Output
    temp_grid: temperature grid; 3d array of floats
    diff_grid: thermal diffusivity grid; 2d array of floats
    '''
    
    # Initialize the temperature grid with zeros
    temp_grid = np.zeros((Nt, Ny, Nx))
    
    # Set initial temperatures for time step 0
    temp_grid[0, :, :x_split+1] = T1  # Material 1
    temp_grid[0, :, x_split+1:] = T2  # Material 2
    
    # Initialize the thermal diffusivity grid
    diff_grid = np.zeros((Ny, Nx))
    
    # Set diffusivity for the two materials
    diff_grid[:, :x_split+1] = alpha1
    diff_grid[:, x_split+1:] = alpha2
    
    return temp_grid, diff_grid



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
        
        # Check if the boundary condition is not on a corner
        # Corners are at (0,0), (0,max), (max,0), (max,max)
        if not ((i == 0 or i == grid.shape[1] - 1) and (j == 0 or j == grid.shape[2] - 1)):
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