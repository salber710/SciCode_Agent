from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from numpy import zeros, linspace, exp, sqrt
import numpy as np



# Background: 
# In computational mathematics, the finite difference method is a numerical technique used to approximate derivatives. 
# A second-order finite difference approximation provides a more accurate estimation of the derivative by considering 
# the values of the function at multiple points. For interior points, the second-order central difference is commonly 
# used: (f(x+delta) - f(x-delta)) / (2*delta). At the boundaries, where central difference cannot be applied symmetrically, 
# a one-sided difference is used: (-3*f(x) + 4*f(x+delta) - f(x+2*delta)) / (2*delta) for forward differences, and 
# (3*f(x) - 4*f(x-delta) + f(x-2*delta)) / (2*delta) for backward differences. These formulas help in calculating 
# the spatial derivatives of a scalar field on a 3D grid, accounting for boundary conditions.

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
    deriv_x = np.zeros_like(fct)
    deriv_y = np.zeros_like(fct)
    deriv_z = np.zeros_like(fct)

    # Compute partial derivatives with respect to x
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if 1 <= i < nx-1:
                    # Central difference for interior points
                    deriv_x[i, j, k] = (fct[i+1, j, k] - fct[i-1, j, k]) / (2 * delta)
                elif i == 0:
                    # Forward difference for the first boundary
                    deriv_x[i, j, k] = (-3*fct[i, j, k] + 4*fct[i+1, j, k] - fct[i+2, j, k]) / (2 * delta)
                else:  # i == nx-1
                    # Backward difference for the last boundary
                    deriv_x[i, j, k] = (3*fct[i, j, k] - 4*fct[i-1, j, k] + fct[i-2, j, k]) / (2 * delta)

    # Compute partial derivatives with respect to y
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if 1 <= j < ny-1:
                    # Central difference for interior points
                    deriv_y[i, j, k] = (fct[i, j+1, k] - fct[i, j-1, k]) / (2 * delta)
                elif j == 0:
                    # Forward difference for the first boundary
                    deriv_y[i, j, k] = (-3*fct[i, j, k] + 4*fct[i, j+1, k] - fct[i, j+2, k]) / (2 * delta)
                else:  # j == ny-1
                    # Backward difference for the last boundary
                    deriv_y[i, j, k] = (3*fct[i, j, k] - 4*fct[i, j-1, k] + fct[i, j-2, k]) / (2 * delta)

    # Compute partial derivatives with respect to z
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if 1 <= k < nz-1:
                    # Central difference for interior points
                    deriv_z[i, j, k] = (fct[i, j, k+1] - fct[i, j, k-1]) / (2 * delta)
                elif k == 0:
                    # Forward difference for the first boundary
                    deriv_z[i, j, k] = (-3*fct[i, j, k] + 4*fct[i, j, k+1] - fct[i, j, k+2]) / (2 * delta)
                else:  # k == nz-1
                    # Backward difference for the last boundary
                    deriv_z[i, j, k] = (3*fct[i, j, k] - 4*fct[i, j, k-1] + fct[i, j, k-2]) / (2 * delta)

    return deriv_x, deriv_y, deriv_z


try:
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

