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

    # Iterate over each boundary condition
    for condition in bc:
        i, j, T = condition
        
        # Apply the boundary condition if it is not a corner point
        if (i > 0 and i < grid.shape[1] - 1) or (j > 0 and j < grid.shape[2] - 1):
            grid[time_index, i, j] = T
    
    return grid


def add_neumann_bc(grid, time_index, bc=np.array([])):
    '''Add Neumann type of boundary conditions to the temperature grid. Users define the real space positions and values of the boundary conditions.
    This function will update boundary conditions constantly as time progresses.
    Boundary conditions will not be applied to corner grid points.
    Input
    grid: the temperature grid for the problem; 3d array of floats
    time_index: the function will update the boundary conditions for the slice with this time axis index; int
    bc: a 2d array where each row has three elements: i, j, and flux. 
        - i and j: the row and column indices to set the boundary conditions; ints
        - flux: the value of the Neumann boundary condition (heat flux); float
    Output
    grid: the updated temperature grid; 3d array of floats
    '''
    
    # Get the shape of the grid
    Ny, Nx = grid.shape[1], grid.shape[2]
    
    # Iterate over each boundary condition
    for condition in bc:
        i, j, flux = condition
        
        # Apply the Neumann boundary condition if it is not a corner point
        if (i > 0 and i < Ny - 1) or (j > 0 and j < Nx - 1):
            # Determine if we're on a boundary and apply the appropriate difference method
            if i == 0:  # Top boundary, use forward difference
                grid[time_index, i, j] = grid[time_index, i+1, j] - flux
            elif i == Ny - 1:  # Bottom boundary, use backward difference
                grid[time_index, i, j] = grid[time_index, i-1, j] + flux
            elif j == 0:  # Left boundary, use forward difference
                grid[time_index, i, j] = grid[time_index, i, j+1] - flux
            elif j == Nx - 1:  # Right boundary, use backward difference
                grid[time_index, i, j] = grid[time_index, i, j-1] + flux
    
    return grid



def heat_equation(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2, bc_dirichlet, bc_neumann):
    '''Main function to numerically solve a 2d heat equation problem.
    Input
    Nt: time dimension of the 3d temperature grid; int
    Nx: x-dimension (number of columns) of the 3d temperature grid; int
    Ny: y-dimension (number of rows) of the 3d temperature grid; int
    x_split: the column index of the vertical interface. All columns up to and including this index (material 1) will have T1 and alpha1, and all columns with a larger index (material 2) will have T2 and alpha2; int
    T1: the initial temperature of each grid point for material 1(in Celsius); float
    alpha1: the thermal diffusivity of material 1; float
    T2: the initial temperature of each grid point for material 2(in Celsius); float
    alpha2: the thermal diffusivity of material 2; float
    bc_dirichlet: a 2d array where each row has three elements: i, j, and T. 
        - i and j: the row and column indices to set the boundary conditions; ints
        - T: the value of the Dirichlet boundary condition; float
    bc_neumann: a 2d array where each row has three elements: i, j, and T. 
        - i and j: the row and column indices to set the boundary conditions; ints
        - T: the value of the Neumann boundary condition; float
    Output
    temp_grid: the temperature grid of the heat equation problem; 3d array of floats
    '''
    
    # Initialize the temperature and diffusivity grids
    temp_grid, diff_grid = init_grid(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2)
    
    # Calculate the time increment using the maximum thermal diffusivity
    dt = 1 / (4 * max(alpha1, alpha2))
    
    # Iterate over each time step
    for t in range(1, Nt):
        # Update the temperature grid using central difference method
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                alpha = diff_grid[i, j]
                temp_grid[t, i, j] = temp_grid[t-1, i, j] + alpha * dt * (
                    (temp_grid[t-1, i+1, j] - 2*temp_grid[t-1, i, j] + temp_grid[t-1, i-1, j]) +
                    (temp_grid[t-1, i, j+1] - 2*temp_grid[t-1, i, j] + temp_grid[t-1, i, j-1])
                )
        
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