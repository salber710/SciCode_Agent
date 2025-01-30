import numpy as np

# Background: The 2D heat equation is a partial differential equation that describes the distribution of heat (or variation in temperature) in a given region over time. In this problem, we are dealing with a grid that represents a physical space divided into two materials with different thermal properties. The thermal diffusivity is a material-specific property that indicates how quickly heat spreads through the material. The task is to initialize a 3D array to store temperature values over time and a 2D array to store thermal diffusivity values for each grid point. The grid is divided by a vertical interface, with different initial temperatures and thermal diffusivities on either side of the interface.

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

    
    # Validate x_split
    if x_split >= Nx or x_split < 0:
        raise IndexError("x_split must be within the range of 0 to Nx-1")
    
    # Handle the case where Nx is zero
    if Nx == 0:
        return np.empty((Nt, Ny, Nx)), np.empty((Ny, Nx))
    
    # Initialize the temperature grid with zeros
    temp_grid = np.zeros((Nt, Ny, Nx), dtype=float)
    
    # Set the initial temperature for the first time step if Nt > 0
    if Nt > 0:
        temp_grid[0, :, :x_split+1] = T1  # Material 1
        temp_grid[0, :, x_split+1:] = T2  # Material 2
    
    # Initialize the thermal diffusivity grid
    diff_grid = np.zeros((Ny, Nx), dtype=float)
    
    # Set the thermal diffusivity for each material
    diff_grid[:, :x_split+1] = alpha1  # Material 1
    diff_grid[:, x_split+1:] = alpha2  # Material 2
    
    return temp_grid, diff_grid


# Background: Dirichlet boundary conditions are a type of boundary condition used in differential equations where the value of the function is specified on the boundary of the domain. In the context of the heat equation, this means setting the temperature at specific grid points to a fixed value. This is useful for simulating scenarios where the temperature at the boundary is controlled or known. When applying these conditions, it is important to avoid setting them at corner points to prevent over-constraining the system, which can lead to numerical instability or inaccuracies.


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
    
    # Get the dimensions of the grid
    Ny, Nx = grid.shape[1], grid.shape[2]
    
    # Iterate over each boundary condition provided
    for condition in bc:
        if len(condition) != 3:
            continue  # Skip invalid boundary condition entries
        
        i, j, T = condition
        
        # Convert indices to integers if they are floats
        i, j = int(i), int(j)
        
        # Check if the indices are within the valid range
        if i < 0 or i >= Ny or j < 0 or j >= Nx:
            continue  # Skip out of bounds indices
        
        # Check if the position is not a corner
        if (i == 0 or i == Ny - 1) and (j == 0 or j == Nx - 1):
            continue  # Skip corner points
        
        # Apply the boundary condition
        grid[time_index, i, j] = T
    
    return grid


# Background: Neumann boundary conditions specify the derivative (gradient) of the function at the boundary rather than the function's value itself. 
# In the context of the heat equation, this means specifying the rate of change of temperature at the boundary, which can represent heat flux. 
# To implement Neumann boundary conditions numerically, we use finite difference methods. 
# For a boundary point, the forward difference method is used if the boundary is at the start of the grid, and the backward difference method is used if the boundary is at the end. 
# This ensures that the derivative is calculated in the direction pointing outward from the boundary. 
# Neumann conditions are not applied to corner points to avoid over-constraining the system.

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
    # Get the dimensions of the grid
    Ny, Nx = grid.shape[1], grid.shape[2]
    
    # Iterate over each boundary condition provided
    for condition in bc:
        if len(condition) != 3:
            continue  # Skip invalid boundary condition entries
        
        i, j, T = condition
        
        # Convert indices to integers if they are floats
        i, j = int(i), int(j)
        
        # Check if the indices are within the valid range
        if i < 0 or i >= Ny or j < 0 or j >= Nx:
            continue  # Skip out of bounds indices
        
        # Check if the position is not a corner
        if (i == 0 or i == Ny - 1) and (j == 0 or j == Nx - 1):
            continue  # Skip corner points
        
        # Apply Neumann boundary condition
        if i == 0:  # Top boundary, use forward difference
            grid[time_index, i, j] = grid[time_index, i + 1, j] - T
        elif i == Ny - 1:  # Bottom boundary, use backward difference
            grid[time_index, i, j] = grid[time_index, i - 1, j] + T
        elif j == 0:  # Left boundary, use forward difference
            grid[time_index, i, j] = grid[time_index, i, j + 1] - T
        elif j == Nx - 1:  # Right boundary, use backward difference
            grid[time_index, i, j] = grid[time_index, i, j - 1] + T
    
    return grid



# Background: The 2D heat equation can be solved numerically using the finite difference method. 
# This involves discretizing the spatial domain into a grid and updating the temperature at each grid point over time.
# The central difference method is used to approximate the second spatial derivatives, which are key in the heat equation.
# The time increment is chosen based on the maximum thermal diffusivity to ensure numerical stability, specifically using 
# the condition Δt = 1 / (4 * max(alpha1, alpha2)). Dirichlet and Neumann boundary conditions are applied at each time step 
# to maintain the specified temperature values and gradients at the boundaries, respectively.


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
    alpha2: the heat conductivity of material 2; float
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
    
    # Calculate the time increment for stability
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

from scicode.parse.parse import process_hdf5_to_tuple
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
