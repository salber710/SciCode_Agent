from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from numpy import zeros, linspace, exp, sqrt
import numpy as np



# Background: 
# In numerical analysis, finite difference methods are used to approximate derivatives of functions.
# For a scalar field f(x, y, z) sampled on a grid with uniform spacing delta, the second-order central finite difference
# approximation for the partial derivative with respect to x is given by:
# ∂f/∂x ≈ (f(x+delta, y, z) - f(x-delta, y, z)) / (2*delta)
# For the boundaries, where central differences cannot be applied symmetrically, a second-order one-sided difference is used.
# For the left boundary (e.g., at x=0), the forward difference is:
# ∂f/∂x ≈ (-3*f(x, y, z) + 4*f(x+delta, y, z) - f(x+2*delta, y, z)) / (2*delta)
# For the right boundary (e.g., at x=nx-1), the backward difference is:
# ∂f/∂x ≈ (3*f(x, y, z) - 4*f(x-delta, y, z) + f(x-2*delta, y, z)) / (2*delta)
# These formulas are applied analogously for the y and z directions.

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

    # Central differences for internal points
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)

    # One-sided differences for boundaries
    # x direction
    deriv_x[0, :, :] = (-3*fct[0, :, :] + 4*fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    deriv_x[-1, :, :] = (3*fct[-1, :, :] - 4*fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)
    # y direction
    deriv_y[:, 0, :] = (-3*fct[:, 0, :] + 4*fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    deriv_y[:, -1, :] = (3*fct[:, -1, :] - 4*fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)
    # z direction
    deriv_z[:, :, 0] = (-3*fct[:, :, 0] + 4*fct[:, :, 1] - fct[:, :, 2]) / (2 * delta)
    deriv_z[:, :, -1] = (3*fct[:, :, -1] - 4*fct[:, :, -2] + fct[:, :, -3]) / (2 * delta)

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

