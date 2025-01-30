from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from numpy import zeros, linspace, exp, sqrt
import numpy as np



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
    deriv_x = np.zeros((nx, ny, nz))
    deriv_y = np.zeros((nx, ny, nz))
    deriv_z = np.zeros((nx, ny, nz))
    
    # Compute internal points using central difference
    # Partial derivative with respect to x
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)

    # Partial derivative with respect to y
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)

    # Partial derivative with respect to z
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)

    # Compute boundary points using one-sided differences
    # x-direction boundaries
    deriv_x[0, :, :] = (-3*fct[0, :, :] + 4*fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    deriv_x[-1, :, :] = (3*fct[-1, :, :] - 4*fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)

    # y-direction boundaries
    deriv_y[:, 0, :] = (-3*fct[:, 0, :] + 4*fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    deriv_y[:, -1, :] = (3*fct[:, -1, :] - 4*fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)

    # z-direction boundaries
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