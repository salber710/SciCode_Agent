import numpy as np

# Background: The 2D heat equation models the distribution of temperature in a given region over time. In this problem,
# we are dealing with a 2D grid representing a spatial domain, with a third dimension representing time. The grid is divided 
# into two materials by a vertical interface at a specified x-coordinate (x_split). Each material has its own initial temperature 
# and thermal diffusivity. The thermal diffusivity is a material property that indicates how quickly heat spreads through the material. 
# In this step, we need to initialize the temperature grid (a 3D array) for storing temperatures over time, and a 2D array for 
# storing the thermal diffusivity of each grid point. The temperature grid should be initialized such that at the first time step,
# grid points belonging to material 1 have the initial temperature T1 and those belonging to material 2 have the initial temperature T2.
# For subsequent time steps, the temperature values should be initialized to zero. The thermal diffusivity grid stores the diffusivity 
# values for each material at each grid point.

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

    
    # Initialize the temperature grid with zeros
    temp_grid = np.zeros((Nt, Ny, Nx))
    
    # Set the initial condition for time step 0
    # Material 1: x <= x_split
    temp_grid[0, :, :x_split+1] = T1
    # Material 2: x > x_split
    temp_grid[0, :, x_split+1:] = T2
    
    # Initialize the diffusivity grid
    diff_grid = np.zeros((Ny, Nx))
    
    # Assign the thermal diffusivity values for each material
    # Material 1: x <= x_split
    diff_grid[:, :x_split+1] = alpha1
    # Material 2: x > x_split
    diff_grid[:, x_split+1:] = alpha2
    
    return temp_grid, diff_grid


# Background: Dirichlet boundary conditions are a type of boundary condition used in partial differential equations,
# including the heat equation. They specify the values that a solution needs to take on the boundary of the domain.
# For a 2D spatial grid, this means specifying temperature values at certain grid points on the boundary. In this
# implementation, boundary conditions are applied to specific grid points at a given time slice, except for the
# corners of the grid. This allows for modeling scenarios where the temperature is fixed at certain boundaries over time.


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
    
    Ny, Nx = grid.shape[1], grid.shape[2]
    
    # Iterate over the boundary conditions provided in the array `bc`
    for boundary in bc:
        i, j, T = boundary
        
        # Skip corners: (0,0), (0, Nx-1), (Ny-1, 0), (Ny-1, Nx-1)
        if (i == 0 and j == 0) or (i == 0 and j == Nx-1) or (i == Ny-1 and j == 0) or (i == Ny-1 and j == Nx-1):
            continue
        
        # Apply the boundary condition to the specified grid point at the given time slice
        grid[time_index, i, j] = T
    
    return grid


# Background: Neumann boundary conditions specify the derivative of a function at the boundary, rather than the function's value as in Dirichlet conditions.
# In the context of the heat equation, this means specifying the rate of change of temperature (or heat flux) at the boundary. The direction of the normal
# to the boundary determines whether a forward or backward finite difference method should be applied to approximate the derivative. 
# Forward differences are used when the normal points out of the boundary at the start of the domain, while backward differences are used at the end.
# This approach allows for the continuous updating of boundary conditions over time, excluding corner grid points, which are typically handled separately.

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

    Ny, Nx = grid.shape[1], grid.shape[2]
    
    # Iterate over the boundary conditions provided in the array `bc`
    for boundary in bc:
        i, j, dT = boundary
        
        # Skip corners: (0,0), (0, Nx-1), (Ny-1, 0), (Ny-1, Nx-1)
        if (i == 0 and j == 0) or (i == 0 and j == Nx-1) or (i == Ny-1 and j == 0) or (i == Ny-1 and j == Nx-1):
            continue
        
        # Apply Neumann boundary conditions
        if i == 0:  # Top boundary, forward difference
            grid[time_index, i, j] = grid[time_index, i+1, j] - dT
        elif i == Ny-1:  # Bottom boundary, backward difference
            grid[time_index, i, j] = grid[time_index, i-1, j] + dT
        elif j == 0:  # Left boundary, forward difference
            grid[time_index, i, j] = grid[time_index, i, j+1] - dT
        elif j == Nx-1:  # Right boundary, backward difference
            grid[time_index, i, j] = grid[time_index, i, j-1] + dT
    
    return grid



# Background: The 2D heat equation can be solved numerically using the finite difference method, which approximates derivatives
# with discrete differences. The central difference method is a popular choice for spatial derivatives, as it provides
# a good balance between accuracy and computational cost. In the context of a 2D grid, central differences are used to
# approximate the Laplacian operator, which describes the flow of heat. Numerical stability is crucial in these computations,
# which is why the time increment (dt) is chosen based on the maximum thermal diffusivity of the materials involved.
# Dirichlet and Neumann boundary conditions must be applied at each time step to ensure the solution respects the problem's
# physical constraints. This setup ensures that heat flow is computed accurately over time, respecting the properties
# of the materials and the imposed boundary conditions.


def heat_equation(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2, bc_dirichlet, bc_neumann):
    '''Main function to numerically solve a 2d heat equation problem.
    Input
    Nt: time dimension of the 3d temperature grid; int
    Nx: x-dimension (number of columns) of the 3d temperature grid; int
    Ny: y-dimension (number of rows) of the 3d temperature grid; int
    x_split: the column index of the vertical interface. All columns up to and including this index (material 1) will have T1 and alpha1, and all columns with a larger index (material 2) will have T2 and alpha2; int
    T1: the initial temperature of each grid point for material 1(in Celsius); float
    alpha1: the thermal diffusivity of material 1; float
    T1: the initial temperature of each grid point for material 2(in Celsius); float
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

    # Initialize the grids
    temp_grid, diff_grid = init_grid(Nt, Nx, Ny, x_split, T1, alpha1, T2, alpha2)
    
    # Calculate time increment for stability
    dt = 1 / (4 * max(alpha1, alpha2))
    
    # Perform time-stepping
    for t in range(1, Nt):
        # Copy the previous time step's temperature grid
        temp_prev = temp_grid[t-1]
        
        # Update temperatures using central difference method
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                alpha = diff_grid[i, j]
                temp_grid[t, i, j] = temp_prev[i, j] + dt * alpha * (
                    (temp_prev[i+1, j] - 2*temp_prev[i, j] + temp_prev[i-1, j]) +
                    (temp_prev[i, j+1] - 2*temp_prev[i, j] + temp_prev[i, j-1])
                )
        
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
