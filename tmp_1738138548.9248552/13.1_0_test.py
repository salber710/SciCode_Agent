from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from numpy import zeros, linspace, exp, sqrt
import numpy as np



# Background: In numerical analysis, the finite difference method is used to approximate derivatives of functions. 
# For a scalar field defined on a 3D grid, the partial derivative with respect to a spatial direction can be 
# approximated using finite differences. A second-order central difference scheme provides a more accurate 
# approximation by considering the function values at points on either side of the point of interest. 
# For a function f(x, y, z), the second-order central difference approximation for the partial derivative 
# with respect to x at a point (i, j, k) is given by:
# ∂f/∂x ≈ (f(i+1, j, k) - f(i-1, j, k)) / (2 * delta)
# At the boundaries, where central differences cannot be applied, a second-order one-sided difference is used:
# Forward difference at the start: ∂f/∂x ≈ (-3*f(i, j, k) + 4*f(i+1, j, k) - f(i+2, j, k)) / (2 * delta)
# Backward difference at the end: ∂f/∂x ≈ (3*f(i, j, k) - 4*f(i-1, j, k) + f(i-2, j, k)) / (2 * delta)
# Similar formulas apply for the y and z directions.

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

    # Compute ∂f/∂x
    for i in range(1, nx-1):
        deriv_x[i, :, :] = (fct[i+1, :, :] - fct[i-1, :, :]) / (2 * delta)
    # One-sided differences for boundaries
    deriv_x[0, :, :] = (-3*fct[0, :, :] + 4*fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    deriv_x[-1, :, :] = (3*fct[-1, :, :] - 4*fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)

    # Compute ∂f/∂y
    for j in range(1, ny-1):
        deriv_y[:, j, :] = (fct[:, j+1, :] - fct[:, j-1, :]) / (2 * delta)
    # One-sided differences for boundaries
    deriv_y[:, 0, :] = (-3*fct[:, 0, :] + 4*fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    deriv_y[:, -1, :] = (3*fct[:, -1, :] - 4*fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)

    # Compute ∂f/∂z
    for k in range(1, nz-1):
        deriv_z[:, :, k] = (fct[:, :, k+1] - fct[:, :, k-1]) / (2 * delta)
    # One-sided differences for boundaries
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e