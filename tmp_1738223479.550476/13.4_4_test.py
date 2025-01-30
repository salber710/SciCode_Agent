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
    # Get the shape of the input field
    nx, ny, nz = fct.shape
    
    # Initialize the Laplacian array with zeros
    lap = np.zeros((nx, ny, nz), dtype=fct.dtype)
    
    # Precompute the reciprocal of delta squared
    inv_delta_sq = 1.0 / (delta * delta)

    # Use numpy's slicing and addition to compute the Laplacian
    slice_x = slice(1, -1)
    slice_y = slice(1, -1)
    slice_z = slice(1, -1)

    # Compute finite differences using numpy's addition and subtraction
    d2fdx2 = (fct[2:, slice_y, slice_z] - 2 * fct[1:-1, slice_y, slice_z] + fct[:-2, slice_y, slice_z])
    d2fdy2 = (fct[slice_x, 2:, slice_z] - 2 * fct[slice_x, 1:-1, slice_z] + fct[slice_x, :-2, slice_z])
    d2fdz2 = (fct[slice_x, slice_y, 2:] - 2 * fct[slice_x, slice_y, 1:-1] + fct[slice_x, slice_y, :-2])

    # Sum the second derivatives
    lap[slice_x, slice_y, slice_z] = (d2fdx2 + d2fdy2 + d2fdz2) * inv_delta_sq
    
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


    # Get the shape of the input field
    nx, ny, nz = fct.shape

    # Initialize gradient arrays with zeros
    grad_x = np.zeros((nx, ny, nz), dtype=fct.dtype)
    grad_y = np.zeros((nx, ny, nz), dtype=fct.dtype)
    grad_z = np.zeros((nx, ny, nz), dtype=fct.dtype)

    # Precompute the reciprocal of delta
    inv_delta = 1.0 / (2.0 * delta)

    # Use NumPy's advanced indexing to vectorize gradient computation
    ix = np.arange(1, nx-1)
    iy = np.arange(1, ny-1)
    iz = np.arange(1, nz-1)
    
    # Compute gradients using broadcasting
    grad_x[ix, :, :] = (fct[ix+1, :, :] - fct[ix-1, :, :]) * inv_delta
    grad_y[:, iy, :] = (fct[:, iy+1, :] - fct[:, iy-1, :]) * inv_delta
    grad_z[:, :, iz] = (fct[:, :, iz+1] - fct[:, :, iz-1]) * inv_delta

    # Zero out the boundaries explicitly
    grad_x[0, :, :] = grad_x[-1, :, :] = 0
    grad_y[:, 0, :] = grad_y[:, -1, :] = 0
    grad_z[:, :, 0] = grad_z[:, :, -1] = 0

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
    
    # Get the shape of the input field components
    nx, ny, nz = v_x.shape

    # Initialize divergence array with zeros
    div = np.zeros((nx, ny, nz), dtype=v_x.dtype)

    # Precompute the reciprocal of delta
    inv_delta = 1.0 / (2.0 * delta)

    # Compute divergence using a staggered approach: process each component in isolation
    # and merge contributions using broadcasting and reshaping for clarity.
    
    # Divergence contribution from v_x
    div_x = np.zeros_like(div)
    div_x[1:-1, :, :] = (v_x[2:, :, :] - v_x[:-2, :, :]) * inv_delta
    
    # Divergence contribution from v_y
    div_y = np.zeros_like(div)
    div_y[:, 1:-1, :] = (v_y[:, 2:, :] - v_y[:, :-2, :]) * inv_delta
    
    # Divergence contribution from v_z
    div_z = np.zeros_like(div)
    div_z[:, :, 1:-1] = (v_z[:, :, 2:] - v_z[:, :, :-2]) * inv_delta

    # Sum all components to get final divergence
    div = div_x + div_y + div_z

    # Boundary values are zero by initialization
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