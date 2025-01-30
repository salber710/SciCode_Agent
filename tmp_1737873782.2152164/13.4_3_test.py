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
    deriv_x = np.zeros_like(fct)
    deriv_y = np.zeros_like(fct)
    deriv_z = np.zeros_like(fct)

    # Compute derivatives using central differences for interior points
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)

    # Compute derivatives using one-sided differences for boundary points
    deriv_x[0, :, :] = (-3 * fct[0, :, :] + 4 * fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    deriv_x[-1, :, :] = (3 * fct[-1, :, :] - 4 * fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)

    deriv_y[:, 0, :] = (-3 * fct[:, 0, :] + 4 * fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    deriv_y[:, -1, :] = (3 * fct[:, -1, :] - 4 * fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)

    deriv_z[:, :, 0] = (-3 * fct[:, :, 0] + 4 * fct[:, :, 1] - fct[:, :, 2]) / (2 * delta)
    deriv_z[:, :, -1] = (3 * fct[:, :, -1] - 4 * fct[:, :, -2] + fct[:, :, -3]) / (2 * delta)

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
    nx, ny, nz = fct.shape
    lap = np.zeros_like(fct)

    # Compute the Laplacian using central differences for interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                lap[i, j, k] = (fct[i+1, j, k] - 2*fct[i, j, k] + fct[i-1, j, k]) / (delta**2) + \
                               (fct[i, j+1, k] - 2*fct[i, j, k] + fct[i, j-1, k]) / (delta**2) + \
                               (fct[i, j, k+1] - 2*fct[i, j, k] + fct[i, j, k-1]) / (delta**2)
    
    return lap


def gradient(fct, delta):
    '''Computes the gradient of a scalar field in the interior of a 3D grid using second-order finite differences.
    Parameters:
    -----------
    fct : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    grad_x : numpy.ndarray
        A 3D array representing the partial derivative of the field with respect to the x direction (∂f/∂x). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    grad_y : numpy.ndarray
        A 3D array representing the partial derivative of the field with respect to the y direction (∂f/∂y). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    grad_z : numpy.ndarray
        A 3D array representing the partial derivative of the field with respect to the z direction (∂f/∂z). 
        Shape: (nx, ny, nz). Boundary values are zeroed out.
    '''
    nx, ny, nz = fct.shape
    grad_x = np.zeros_like(fct)
    grad_y = np.zeros_like(fct)
    grad_z = np.zeros_like(fct)

    # Compute gradients using central differences for interior points
    grad_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    grad_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    grad_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)

    # Set boundary values to zero
    grad_x[0, :, :] = 0
    grad_x[-1, :, :] = 0

    grad_y[:, 0, :] = 0
    grad_y[:, -1, :] = 0

    grad_z[:, :, 0] = 0
    grad_z[:, :, -1] = 0

    return grad_x, grad_y, grad_z



def divergence(v_x, v_y, v_z, delta):
    '''Computes the divergence of a 3D vector field using second-order finite differences.
    Parameters:
    -----------
    v_x : numpy.ndarray
        A 3D array representing the x-component of the vector field. Shape: (nx, ny, nz).
    v_y : numpy.ndarray
        A 3D array representing the y-component of the vector field. Shape: (nx, ny, nz).
    v_z : numpy.ndarray
        A 3D array representing the z-component of the vector field. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    div : numpy.ndarray
        A 3D array representing the divergence of the vector field. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    '''
    nx, ny, nz = v_x.shape
    div = np.zeros_like(v_x)

    # Compute divergence using central differences for interior points
    div[1:-1, 1:-1, 1:-1] = (
        (v_x[2:, 1:-1, 1:-1] - v_x[:-2, 1:-1, 1:-1]) / (2 * delta) +
        (v_y[1:-1, 2:, 1:-1] - v_y[1:-1, :-2, 1:-1]) / (2 * delta) +
        (v_z[1:-1, 1:-1, 2:] - v_z[1:-1, 1:-1, :-2]) / (2 * delta)
    )

    # Boundary values are already set to zero by initialization

    return div


try:
    targets = process_hdf5_to_tuple('13.4', 3)
    target = targets[0]
    x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
    fx    = -y
    fy    = x
    fz    = np.zeros(z.shape)
    delta = x[0][1][0] - x[0][0][0]
    assert np.allclose(divergence(fx, fy, fz, delta), target)

    target = targets[1]
    x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
    fx    = x
    fy    = y
    fz    = z
    delta = x[0][1][0] - x[0][0][0]
    assert np.allclose(divergence(fx, fy, fz, delta), target)

    target = targets[2]
    x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
    fx    = np.sin(x*y*z)
    fy    = np.cos(x*y*z)
    fz    = fx*fy
    delta = x[0][1][0] - x[0][0][0]
    assert np.allclose(divergence(fx, fy, fz, delta), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e