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

    # Use numpy convolution to compute differences
    kernel_central = np.array([-1, 0, 1]) / (2 * delta)
    kernel_one_sided = np.array([-3, 4, -1]) / (2 * delta)

    # Central differences for interior points
    for i in range(1, nx-1):
        deriv_x[i, :, :] = np.convolve(fct[i-1:i+2, :, :].reshape(-1), kernel_central, mode='valid').reshape(ny, nz)
    for j in range(1, ny-1):
        deriv_y[:, j, :] = np.convolve(fct[:, j-1:j+2, :].reshape(-1), kernel_central, mode='valid').reshape(nx, nz)
    for k in range(1, nz-1):
        deriv_z[:, :, k] = np.convolve(fct[:, :, k-1:k+2].reshape(-1), kernel_central, mode='valid').reshape(nx, ny)

    # One-sided differences for boundaries
    # x boundaries
    for i in [0, nx-1]:
        deriv_x[i, :, :] = (np.convolve(fct[max(0, i-1):i+2, :, :].reshape(-1), kernel_one_sided, mode='valid')).reshape(ny, nz) if i == 0 else (-np.convolve(fct[i-2:i+1, :, :].reshape(-1), kernel_one_sided[::-1], mode='valid')).reshape(ny, nz)

    # y boundaries
    for j in [0, ny-1]:
        deriv_y[:, j, :] = (np.convolve(fct[:, max(0, j-1):j+2, :].reshape(-1), kernel_one_sided, mode='valid')).reshape(nx, nz) if j == 0 else (-np.convolve(fct[:, j-2:j+1, :].reshape(-1), kernel_one_sided[::-1], mode='valid')).reshape(nx, nz)

    # z boundaries
    for k in [0, nz-1]:
        deriv_z[:, :, k] = (np.convolve(fct[:, :, max(0, k-1):k+2].reshape(-1), kernel_one_sided, mode='valid')).reshape(nx, ny) if k == 0 else (-np.convolve(fct[:, :, k-2:k+1].reshape(-1), kernel_one_sided[::-1], mode='valid')).reshape(nx, ny)

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