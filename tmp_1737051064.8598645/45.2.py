import numpy as np

# Background: The 2D heat equation is a partial differential equation that describes the distribution of heat (or variation in temperature) in a given region over time. In this problem, we are dealing with a grid that represents a physical space divided into two materials with different thermal properties. The 3D array will store temperature values over time and space, while the 2D array will store thermal diffusivity values for each spatial point. The thermal diffusivity is a material property that indicates how quickly heat spreads through the material. The grid is initialized with specific temperatures for each material, and the thermal diffusivity is set according to the material type.


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
    alpha2: the thermal diffusivity of material 2; float
    Output
    temp_grid: temperature grid; 3d array of floats
    diff_grid: thermal diffusivity grid; 2d array of floats
    '''

    # Initialize the temperature grid with zeros
    temp_grid = np.zeros((Nt, Ny, Nx), dtype=float)
    
    # Set the initial temperature for the first time step
    temp_grid[0, :, :x_split+1] = T1  # Material 1
    temp_grid[0, :, x_split+1:] = T2  # Material 2
    
    # Initialize the thermal diffusivity grid
    diff_grid = np.zeros((Ny, Nx), dtype=float)
    
    # Set the thermal diffusivity for each material
    diff_grid[:, :x_split+1] = alpha1  # Material 1
    diff_grid[:, x_split+1:] = alpha2  # Material 2
    
    return temp_grid, diff_grid



# Background: Dirichlet boundary conditions are a type of boundary condition used in differential equations where the value of the function is specified on the boundary of the domain. In the context of the heat equation, this means setting the temperature at specific points on the boundary of the grid to a fixed value. This is important for simulating scenarios where the temperature at the boundary is controlled or known. The function will update the temperature grid at specified boundary points for a given time slice, ensuring that these conditions are maintained as the simulation progresses. It is important to note that corner points are excluded from these updates to avoid potential conflicts in boundary condition specifications.


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
    
    # Iterate over each boundary condition specified in the bc array
    for condition in bc:
        i, j, T = condition
        
        # Check if the point is not a corner point
        if not ((i == 0 or i == grid.shape[1] - 1) and (j == 0 or j == grid.shape[2] - 1)):
            # Apply the boundary condition to the specified point
            grid[time_index, i, j] = T
    
    return grid


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('45.2', 3)
target = targets[0]

assert np.allclose(add_dirichlet_bc(init_grid(2, 10, 5, 2, 100, 1, 50, 2)[0], 0, [[1, 0, 20]]), target)
target = targets[1]

assert np.allclose(add_dirichlet_bc(init_grid(2, 10, 5, 2, 100, 1, 50, 2)[0], 0, np.array([[0, j, 40] for j in range(0, 10)])), target)
target = targets[2]

assert np.allclose(add_dirichlet_bc(init_grid(2, 10, 5, 2, 10, 1, 20, 2)[0], 0, np.array([[i, 0, 100] for i in range(0, 5)])), target)
