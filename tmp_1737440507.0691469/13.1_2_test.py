from numpy import zeros, linspace, exp, sqrt
import numpy as np



# Background: The task involves calculating the partial derivatives of a scalar field represented on a 3D grid. 
# Partial derivatives can be approximated using finite difference methods. Here, we use a second-order central 
# difference method for interior points and a one-sided second-order difference method for boundary points. 
# The second-order central difference formula for the derivative of a function f at a point i is given by:
# (f(i+1) - f(i-1)) / (2 * delta), where delta is the grid spacing. For the boundaries, where central differences 
# are not possible, a second-order one-sided difference is used. For example, at the beginning of the array, 
# the derivative is approximated by (-3*f(i) + 4*f(i+1) - f(i+2)) / (2 * delta).

def partial_derivs_vec(fct, delta):
    '''Computes the partial derivatives of a scalar field in three dimensions using second-order finite differences.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in each spatial direction.
    Returns:
    --------
    deriv_x : numpy.ndarray
        The partial derivative of the field with respect to the x direction (∂f/∂x).
    deriv_y : numpy.ndarray
        The partial derivative of the field with respect to the y direction (∂f/∂y).
    deriv_z : numpy.ndarray
        The partial derivative of the field with respect to the z direction (∂f/∂z).
    '''
    nx, ny, nz = fct.shape
    deriv_x = zeros((nx, ny, nz))
    deriv_y = zeros((nx, ny, nz))
    deriv_z = zeros((nx, ny, nz))
    
    # Calculate ∂f/∂x
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i == 0:  # Forward difference at the boundary
                    deriv_x[i, j, k] = (-3*fct[i, j, k] + 4*fct[i+1, j, k] - fct[i+2, j, k]) / (2 * delta)
                elif i == nx - 1:  # Backward difference at the boundary
                    deriv_x[i, j, k] = (3*fct[i, j, k] - 4*fct[i-1, j, k] + fct[i-2, j, k]) / (2 * delta)
                else:  # Central difference
                    deriv_x[i, j, k] = (fct[i+1, j, k] - fct[i-1, j, k]) / (2 * delta)
    
    # Calculate ∂f/∂y
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if j == 0:  # Forward difference at the boundary
                    deriv_y[i, j, k] = (-3*fct[i, j, k] + 4*fct[i, j+1, k] - fct[i, j+2, k]) / (2 * delta)
                elif j == ny - 1:  # Backward difference at the boundary
                    deriv_y[i, j, k] = (3*fct[i, j, k] - 4*fct[i, j-1, k] + fct[i, j-2, k]) / (2 * delta)
                else:  # Central difference
                    deriv_y[i, j, k] = (fct[i, j+1, k] - fct[i, j-1, k]) / (2 * delta)
    
    # Calculate ∂f/∂z
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if k == 0:  # Forward difference at the boundary
                    deriv_z[i, j, k] = (-3*fct[i, j, k] + 4*fct[i, j, k+1] - fct[i, j, k+2]) / (2 * delta)
                elif k == nz - 1:  # Backward difference at the boundary
                    deriv_z[i, j, k] = (3*fct[i, j, k] - 4*fct[i, j, k-1] + fct[i, j, k-2]) / (2 * delta)
                else:  # Central difference
                    deriv_z[i, j, k] = (fct[i, j, k+1] - fct[i, j, k-1]) / (2 * delta)
    
    return deriv_x, deriv_y, deriv_z

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('13.1', 3)
target = targets[0]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
fct   = np.sin(x)*np.sin(2*y)*np.sin(3*z) 
delta = x[0][1][0] - x[0][0][0]
assert np.allclose(partial_derivs_vec(fct, delta), target)
target = targets[1]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
fct   = (x+y)**2+z
delta = x[0][1][0] - x[0][0][0]
assert np.allclose(partial_derivs_vec(fct, delta), target)
target = targets[2]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
fct   = x*z + y*z + x*y
delta = x[0][1][0] - x[0][0][0]
assert np.allclose(partial_derivs_vec(fct, delta), target)
