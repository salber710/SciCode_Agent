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



# Background: The Laplacian operator, denoted as ∇², is a second-order differential operator in n-dimensional Euclidean space,
# defined as the divergence of the gradient of a function. In three-dimensional Cartesian coordinates, the Laplacian of a scalar
# field f(x, y, z) is given by ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z². In numerical analysis, the Laplacian can be approximated using
# finite difference methods. For a 3D grid, a second-order central finite difference scheme is used to approximate the second
# derivatives. The boundary values are set to zero to focus on the interior grid points, ensuring that the Laplacian is only
# calculated for these points.

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

    # Compute the Laplacian in the interior of the grid
    lap[1:-1, 1:-1, 1:-1] = (
        (fct[2:, 1:-1, 1:-1] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[:-2, 1:-1, 1:-1]) / delta**2 +
        (fct[1:-1, 2:, 1:-1] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[1:-1, :-2, 1:-1]) / delta**2 +
        (fct[1:-1, 1:-1, 2:] - 2 * fct[1:-1, 1:-1, 1:-1] + fct[1:-1, 1:-1, :-2]) / delta**2
    )

    # Boundary values are already set to zero by np.zeros_like
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