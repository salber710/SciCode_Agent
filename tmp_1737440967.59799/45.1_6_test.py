import numpy as np



# Background: The heat equation is a partial differential equation that describes how the distribution of heat evolves over time in a solid medium. In a 2D heat equation problem, the medium is represented by a grid, where each point has a temperature value that changes over time based on the thermal diffusivity of the material at that point. Thermal diffusivity is a measure of how quickly heat spreads through a material and is denoted by alpha (α). In this problem, we have a 3D array to store temperature values, where the first dimension represents time, and the second and third dimensions represent spatial coordinates on the grid. We also have a 2D array to store the thermal diffusivity values for each spatial grid point. The grid is split vertically at a specified index, which separates two materials with different initial temperatures and thermal diffusivities. The task is to initialize these arrays with the appropriate values.

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

    
    # Initialize the temperature grid with zeros for all time steps
    temp_grid = np.zeros((Nt, Ny, Nx), dtype=float)
    
    # Set the initial temperature at t=0
    temp_grid[0, :, :x_split+1] = T1  # Material 1
    temp_grid[0, :, x_split+1:] = T2  # Material 2
    
    # Initialize the diffusivity grid
    diff_grid = np.zeros((Ny, Nx), dtype=float)
    
    # Set the diffusivity for each material
    diff_grid[:, :x_split+1] = alpha1  # Material 1
    diff_grid[:, x_split+1:] = alpha2  # Material 2
    
    return temp_grid, diff_grid

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('45.1', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
assert cmp_tuple_or_list(init_grid(2, 10, 5, 2, 100, 1, 50, 2), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
assert cmp_tuple_or_list(init_grid(3, 3, 3, 1, 100, 1, 50, 2), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
assert cmp_tuple_or_list(init_grid(3, 5, 5, 1, 100, 1, 50, 2), target)
