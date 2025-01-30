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

    # Compute interior points using a compact stencil
    for i in range(1, nx-1):
        for j in range(ny):
            for k in range(nz):
                deriv_x[i, j, k] = (fct[i+1, j, k] - fct[i-1, j, k]) / (2 * delta)

    for i in range(nx):
        for j in range(1, ny-1):
            for k in range(nz):
                deriv_y[i, j, k] = (fct[i, j+1, k] - fct[i, j-1, k]) / (2 * delta)

    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz-1):
                deriv_z[i, j, k] = (fct[i, j, k+1] - fct[i, j, k-1]) / (2 * delta)

    # Boundary points using a finite-volume inspired approach
    # x boundaries
    for j in range(ny):
        for k in range(nz):
            deriv_x[0, j, k] = (-3 * fct[0, j, k] + 4 * fct[1, j, k] - fct[2, j, k]) / (2 * delta)
            deriv_x[-1, j, k] = (3 * fct[-1, j, k] - 4 * fct[-2, j, k] + fct[-3, j, k]) / (2 * delta)

    # y boundaries
    for i in range(nx):
        for k in range(nz):
            deriv_y[i, 0, k] = (-3 * fct[i, 0, k] + 4 * fct[i, 1, k] - fct[i, 2, k]) / (2 * delta)
            deriv_y[i, -1, k] = (3 * fct[i, -1, k] - 4 * fct[i, -2, k] + fct[i, -3, k]) / (2 * delta)

    # z boundaries
    for i in range(nx):
        for j in range(ny):
            deriv_z[i, j, 0] = (-3 * fct[i, j, 0] + 4 * fct[i, j, 1] - fct[i, j, 2]) / (2 * delta)
            deriv_z[i, j, -1] = (3 * fct[i, j, -1] - 4 * fct[i, j, -2] + fct[i, j, -3]) / (2 * delta)

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