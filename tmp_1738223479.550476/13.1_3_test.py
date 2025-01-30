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

    deriv_x = np.empty((nx, ny, nz))
    deriv_y = np.empty((nx, ny, nz))
    deriv_z = np.empty((nx, ny, nz))

    # Using gradient and finite difference coefficients
    coeffs_central = np.array([-0.5, 0, 0.5]) / delta
    coeffs_forward = np.array([-1.5, 2, -0.5]) / delta
    coeffs_backward = np.array([0.5, -2, 1.5]) / delta

    # Central differences for interior points
    deriv_x[1:-1, :, :] = np.tensordot(fct, coeffs_central, axes=([0], [0]))[1:-1, :, :]
    deriv_y[:, 1:-1, :] = np.tensordot(fct, coeffs_central, axes=([1], [0]))[:, 1:-1, :]
    deriv_z[:, :, 1:-1] = np.tensordot(fct, coeffs_central, axes=([2], [0]))[:, :, 1:-1]

    # Forward differences for boundaries
    deriv_x[0, :, :] = np.dot(fct[:3, :, :].reshape(3, -1).T, coeffs_forward).reshape(ny, nz)
    deriv_y[:, 0, :] = np.dot(fct[:, :3, :].reshape(nx, 3, nz).transpose(0, 2, 1).reshape(-1, 3), coeffs_forward).reshape(nx, nz)
    deriv_z[:, :, 0] = np.dot(fct[:, :, :3].reshape(nx, ny, 3).transpose(0, 1, 2).reshape(-1, 3), coeffs_forward).reshape(nx, ny)

    # Backward differences for boundaries
    deriv_x[-1, :, :] = np.dot(fct[-3:, :, :].reshape(3, -1).T, coeffs_backward).reshape(ny, nz)
    deriv_y[:, -1, :] = np.dot(fct[:, -3:, :].reshape(nx, 3, nz).transpose(0, 2, 1).reshape(-1, 3), coeffs_backward).reshape(nx, nz)
    deriv_z[:, :, -1] = np.dot(fct[:, :, -3:].reshape(nx, ny, 3).transpose(0, 1, 2).reshape(-1, 3), coeffs_backward).reshape(nx, ny)

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