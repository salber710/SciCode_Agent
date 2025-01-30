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
    T1: the initial temperature of each grid point for material 2(in Celsius); float
    alpha2: the heat conductivity of material 2; float
    Output
    temp_grid: temperature grid; 3d array of floats
    diff_grid: thermal diffusivity grid; 2d array of floats
    '''
    
    # Use a single list comprehension to initialize temp_grid with the first time step setup
    temp_grid = [[[
        T1 if x <= x_split else T2 if t == 0 else 0.0
        for x in range(Nx)
    ] for y in range(Ny)] for t in range(Nt)]
    
    # Use a differing approach for initializing diff_grid using a set default function
    diff_grid = [[0.0] * Nx for _ in range(Ny)]
    for y in range(Ny):
        for x in range(Nx):
            diff_grid[y][x] = alpha1 if x <= x_split else alpha2

    return temp_grid, diff_grid


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e