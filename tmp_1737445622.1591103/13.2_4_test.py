from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

from numpy import zeros, linspace, exp, sqrt
import numpy as np

# Background: Computing partial derivatives on a discrete grid is a common task in numerical simulations.
# The second-order finite difference method approximates the derivative of a function by using a central
# difference scheme. For a function f(x) sampled at discrete points with spacing delta, the derivative
# at a point is approximated as:
#   f'(x) ≈ (f(x+delta) - f(x-delta)) / (2*delta)
# At the boundaries, where central differencing isn't possible, a one-sided finite difference is used:
#   f'(x) ≈ (-3*f(x) + 4*f(x+delta) - f(x+2*delta)) / (2*delta)
# This ensures that the approximation remains second-order accurate while handling edge cases.

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
    deriv_x = zeros((nx, ny, nz))
    deriv_y = zeros((nx, ny, nz))
    deriv_z = zeros((nx, ny, nz))

    # Compute interior derivatives using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                deriv_x[i, j, k] = (fct[i+1, j, k] - fct[i-1, j, k]) / (2 * delta)
                deriv_y[i, j, k] = (fct[i, j+1, k] - fct[i, j-1, k]) / (2 * delta)
                deriv_z[i, j, k] = (fct[i, j, k+1] - fct[i, j, k-1]) / (2 * delta)

    # Compute boundary derivatives using one-sided differences
    # X boundaries
    for j in range(ny):
        for k in range(nz):
            deriv_x[0, j, k] = (-3*fct[0, j, k] + 4*fct[1, j, k] - fct[2, j, k]) / (2 * delta)
            deriv_x[nx-1, j, k] = (3*fct[nx-1, j, k] - 4*fct[nx-2, j, k] + fct[nx-3, j, k]) / (2 * delta)
    
    # Y boundaries
    for i in range(nx):
        for k in range(nz):
            deriv_y[i, 0, k] = (-3*fct[i, 0, k] + 4*fct[i, 1, k] - fct[i, 2, k]) / (2 * delta)
            deriv_y[i, ny-1, k] = (3*fct[i, ny-1, k] - 4*fct[i, ny-2, k] + fct[i, ny-3, k]) / (2 * delta)
    
    # Z boundaries
    for i in range(nx):
        for j in range(ny):
            deriv_z[i, j, 0] = (-3*fct[i, j, 0] + 4*fct[i, j, 1] - fct[i, j, 2]) / (2 * delta)
            deriv_z[i, j, nz-1] = (3*fct[i, j, nz-1] - 4*fct[i, j, nz-2] + fct[i, j, nz-3]) / (2 * delta)

    return deriv_x, deriv_y, deriv_z



# Background: The Laplacian operator is a second-order differential operator in n-dimensional Euclidean space,
# defined as the divergence of the gradient of a function. In 3D, it is often written as ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².
# It is used to measure the rate at which the average value of a function around a point differs from the value at that point.
# In numerical simulations, we approximate the Laplacian using finite differences. For a 3D grid, the Laplacian can be computed
# using central finite differences for the interior points. At the boundaries, we set the Laplacian to zero as specified.


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
    lap = zeros((nx, ny, nz))
    
    # Compute interior Laplacian using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                lap[i, j, k] = (
                    (fct[i+1, j, k] - 2*fct[i, j, k] + fct[i-1, j, k]) / (delta**2) +
                    (fct[i, j+1, k] - 2*fct[i, j, k] + fct[i, j-1, k]) / (delta**2) +
                    (fct[i, j, k+1] - 2*fct[i, j, k] + fct[i, j, k-1]) / (delta**2)
                )

    # Boundary values are already set to zero by initialization with zeros
    
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

