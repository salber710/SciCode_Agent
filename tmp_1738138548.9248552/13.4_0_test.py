from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from numpy import zeros, linspace, exp, sqrt
import numpy as np


def partial_derivs_vec(fct, delta):
    nx, ny, nz = fct.shape
    deriv_x = np.zeros_like(fct)
    deriv_y = np.zeros_like(fct)
    deriv_z = np.zeros_like(fct)

    # Compute ∂f/∂x
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    deriv_x[0, :, :] = (-3 * fct[0, :, :] + 4 * fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    deriv_x[-1, :, :] = (3 * fct[-1, :, :] - 4 * fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)

    # Compute ∂f/∂y
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    deriv_y[:, 0, :] = (-3 * fct[:, 0, :] + 4 * fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    deriv_y[:, -1, :] = (3 * fct[:, -1, :] - 4 * fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)

    # Compute ∂f/∂z
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)
    deriv_z[:, :, 0] = (-3 * fct[:, :, 0] + 4 * fct[:, :, 1] - fct[:, :, 2]) / (2 * delta)
    deriv_z[:, :, -1] = (3 * fct[:, :, -1] - 4 * fct[:, :, -2] + fct[:, :, -3]) / (2 * delta)

    return deriv_x, deriv_y, deriv_z



def laplace(fct, delta):
    nx, ny, nz = fct.shape
    lap = np.zeros_like(fct)
    delta_sq = delta ** 2

    # Compute the Laplacian using a combination of slicing and numpy pad for boundary handling
    padded_fct = np.pad(fct, pad_width=1, mode='constant', constant_values=0)
    lap[1:-1, 1:-1, 1:-1] = (
        padded_fct[2:, 1:-1, 1:-1] + padded_fct[:-2, 1:-1, 1:-1] +
        padded_fct[1:-1, 2:, 1:-1] + padded_fct[1:-1, :-2, 1:-1] +
        padded_fct[1:-1, 1:-1, 2:] + padded_fct[1:-1, 1:-1, :-2] -
        6 * padded_fct[1:-1, 1:-1, 1:-1]
    ) / delta_sq

    return lap



def gradient(fct, delta):
    nx, ny, nz = fct.shape
    grad_x = np.zeros_like(fct)
    grad_y = np.zeros_like(fct)
    grad_z = np.zeros_like(fct)

    # Compute gradients using second-order central differences
    # Use np.roll for circular shift and handle boundaries separately
    fct_x_plus = np.roll(fct, -1, axis=0)
    fct_x_minus = np.roll(fct, 1, axis=0)
    fct_y_plus = np.roll(fct, -1, axis=1)
    fct_y_minus = np.roll(fct, 1, axis=1)
    fct_z_plus = np.roll(fct, -1, axis=2)
    fct_z_minus = np.roll(fct, 1, axis=2)

    grad_x[1:-1, :, :] = (fct_x_plus[1:-1, :, :] - fct_x_minus[1:-1, :, :]) / (2 * delta)
    grad_y[:, 1:-1, :] = (fct_y_plus[:, 1:-1, :] - fct_y_minus[:, 1:-1, :]) / (2 * delta)
    grad_z[:, :, 1:-1] = (fct_z_plus[:, :, 1:-1] - fct_z_minus[:, :, 1:-1]) / (2 * delta)

    # Zero out the boundary values explicitly
    grad_x[0, :, :] = grad_x[-1, :, :] = 0
    grad_y[:, 0, :] = grad_y[:, -1, :] = 0
    grad_z[:, :, 0] = grad_z[:, :, -1] = 0

    return grad_x, grad_y, grad_z



# Background: The divergence of a vector field is a scalar field that represents the net rate of flow of the vector field out of an infinitesimal volume. In the context of a 3D vector field, the divergence is calculated as the sum of the partial derivatives of each component of the vector field with respect to its corresponding spatial dimension. For a vector field v = (v_x, v_y, v_z), the divergence is given by ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z. In this implementation, we use second-order finite differences to approximate these partial derivatives on a 3D mesh. The boundary values are set to zero to ensure that the divergence is only computed for the interior points of the mesh.

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

    # Compute the divergence using second-order central differences
    # ∂v_x/∂x
    div[1:-1, :, :] += (v_x[2:, :, :] - v_x[:-2, :, :]) / (2 * delta)
    # ∂v_y/∂y
    div[:, 1:-1, :] += (v_y[:, 2:, :] - v_y[:, :-2, :]) / (2 * delta)
    # ∂v_z/∂z
    div[:, :, 1:-1] += (v_z[:, :, 2:] - v_z[:, :, :-2]) / (2 * delta)

    # Zero out the boundary values explicitly
    div[0, :, :] = div[-1, :, :] = 0
    div[:, 0, :] = div[:, -1, :] = 0
    div[:, :, 0] = div[:, :, -1] = 0

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