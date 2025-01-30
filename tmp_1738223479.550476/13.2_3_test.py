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

    # Define the central difference kernel
    central_kernel = np.array([-0.5, 0, 0.5]) / delta
    # Define the one-sided difference kernel
    forward_kernel = np.array([-1.5, 2, -0.5]) / delta
    backward_kernel = np.array([0.5, -2, 1.5]) / delta

    # Initialize the derivative arrays
    deriv_x = np.zeros_like(fct)
    deriv_y = np.zeros_like(fct)
    deriv_z = np.zeros_like(fct)

    # Compute interior derivatives using central differences
    deriv_x[1:-1, :, :] = convolve1d(fct, central_kernel, axis=0)[1:-1, :, :]
    deriv_y[:, 1:-1, :] = convolve1d(fct, central_kernel, axis=1)[:, 1:-1, :]
    deriv_z[:, :, 1:-1] = convolve1d(fct, central_kernel, axis=2)[:, :, 1:-1]

    # Compute boundary derivatives using one-sided differences
    deriv_x[0, :, :] = convolve1d(fct[:3, :, :], forward_kernel, axis=0)[0, :, :]
    deriv_x[-1, :, :] = convolve1d(fct[-3:, :, :], backward_kernel, axis=0)[-1, :, :]

    deriv_y[:, 0, :] = convolve1d(fct[:, :3, :], forward_kernel, axis=1)[:, 0, :]
    deriv_y[:, -1, :] = convolve1d(fct[:, -3:, :], backward_kernel, axis=1)[:, -1, :]

    deriv_z[:, :, 0] = convolve1d(fct[:, :, :3], forward_kernel, axis=2)[:, :, 0]
    deriv_z[:, :, -1] = convolve1d(fct[:, :, -3:], backward_kernel, axis=2)[:, :, -1]

    return deriv_x, deriv_y, deriv_z




def laplace(fct, delta):
    '''Computes the Laplacian of a scalar field in the interior of a 3D grid using second-order finite differences.
    This function calculates the Laplacian of a scalar field on a structured 3D grid using a central finite difference
    scheme. The output boundary values are set to zero to ensure the Laplacian is only calculated for the interior grid points.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in each spatial direction.
    Returns:
    --------
    lap : numpy.ndarray
        A 3D array representing the Laplacian of the scalar field. Shape: (nx, ny, nz).
        The boundary values are set to zero, while the interior values are computed using the finite difference method.
    '''
    # Create a padded array to handle boundaries more easily
    padded_fct = np.pad(fct, pad_width=1, mode='constant', constant_values=0)
    
    # Determine the shape of the input field
    nx, ny, nz = fct.shape
    
    # Initialize the Laplacian array
    lap = np.zeros((nx, ny, nz), dtype=fct.dtype)
    
    # Precompute the finite difference coefficient
    coeff = 1 / (delta ** 2)
    
    # Compute the Laplacian using the padded array for interior points
    # This avoids boundary checks and allows simple indexing
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                lap[i - 1, j - 1, k - 1] = (
                    (padded_fct[i + 1, j, k] - 2 * padded_fct[i, j, k] + padded_fct[i - 1, j, k]) +
                    (padded_fct[i, j + 1, k] - 2 * padded_fct[i, j, k] + padded_fct[i, j - 1, k]) +
                    (padded_fct[i, j, k + 1] - 2 * padded_fct[i, j, k] + padded_fct[i, j, k - 1])
                ) * coeff
    
    return lap


try:
    targets = process_hdf5_to_tuple('13.2', 3)
    target = targets[0]
    x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
    fct   = np.sin(x)*np.sin(2*y)*np.sin(3*z) 
    delta = x[0][1][0] - x[0][0][0]
    assert np.allclose(laplace(fct, delta), target)

    target = targets[1]
    x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
    fct   = x**2+y**2+z**2
    delta = x[0][1][0] - x[0][0][0]
    assert np.allclose(laplace(fct, delta), target)

    target = targets[2]
    x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
    fct   = x**4 + y**2 + z**5
    delta = x[0][1][0] - x[0][0][0]
    assert np.allclose(laplace(fct, delta), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e