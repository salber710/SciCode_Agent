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


# Background: The Laplacian is a measure of the second spatial derivative, which is often used in physics and
# engineering to describe phenomena such as heat conduction, wave propagation, and fluid dynamics. In a discrete
# grid, the Laplacian is typically calculated using finite difference methods. For a function f(x, y, z) sampled
# on a 3D grid with constant spacing delta, the Laplacian can be approximated as:
#   ∇²f ≈ (f(x+delta, y, z) + f(x-delta, y, z) - 2*f(x, y, z)) / delta²
#        + (f(x, y+delta, z) + f(x, y-delta, z) - 2*f(x, y, z)) / delta²
#        + (f(x, y, z+delta) + f(x, y, z-delta) - 2*f(x, y, z)) / delta²
# This is a second-order central finite difference approximation. The boundary values are set to zero to focus
# on accurately computing the Laplacian within the interior of the grid.

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

    # Compute the Laplacian for the interior points using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                lap[i, j, k] = (
                    (fct[i+1, j, k] + fct[i-1, j, k] - 2*fct[i, j, k]) +
                    (fct[i, j+1, k] + fct[i, j-1, k] - 2*fct[i, j, k]) +
                    (fct[i, j, k+1] + fct[i, j, k-1] - 2*fct[i, j, k])
                ) / (delta**2)

    # The boundary values are already zero-initialized by zeros, so no need to set them explicitly.
    
    return lap


# Background: The gradient of a scalar field is a vector field that represents the rate and direction of change
# of the scalar field. In three-dimensional space, the gradient is composed of the partial derivatives of the
# scalar field with respect to each spatial coordinate. The gradient is a fundamental concept in vector calculus
# and is widely used in physics and engineering, particularly in the context of electromagnetism and fluid dynamics.
# For a scalar field f(x, y, z) on a discrete 3D grid with spacing delta, the gradient can be approximated using
# second-order central finite differences. At each interior grid point, the partial derivatives are computed as:
#   ∂f/∂x ≈ (f(x+delta, y, z) - f(x-delta, y, z)) / (2*delta),
#   ∂f/∂y ≈ (f(x, y+delta, z) - f(x, y-delta, z)) / (2*delta),
#   ∂f/∂z ≈ (f(x, y, z+delta) - f(x, y, z-delta)) / (2*delta).
# The boundary values of the gradient are set to zero since the derivatives are not well-defined at the boundaries
# using central differences.

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
    grad_x = zeros((nx, ny, nz))
    grad_y = zeros((nx, ny, nz))
    grad_z = zeros((nx, ny, nz))
    
    # Compute the gradient for the interior points using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                grad_x[i, j, k] = (fct[i+1, j, k] - fct[i-1, j, k]) / (2 * delta)
                grad_y[i, j, k] = (fct[i, j+1, k] - fct[i, j-1, k]) / (2 * delta)
                grad_z[i, j, k] = (fct[i, j, k+1] - fct[i, j, k-1]) / (2 * delta)

    # The boundary values are already zero-initialized by zeros, so no need to set them explicitly.

    return grad_x, grad_y, grad_z


# Background: In vector calculus, the divergence of a vector field is a scalar field that represents
# the net rate of flow of the vector field out of an infinitesimal volume around a given point. It is
# a measure of the "source" or "sink" strength at a point in the field. For a three-dimensional vector
# field v(x, y, z) with components (v_x, v_y, v_z), the divergence is defined as:
#   ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z.
# On a discrete 3D grid with constant spacing delta, the divergence can be approximated using
# second-order central finite differences:
#   ∂v_x/∂x ≈ (v_x(i+1, j, k) - v_x(i-1, j, k)) / (2*delta),
#   ∂v_y/∂y ≈ (v_y(i, j+1, k) - v_y(i, j-1, k)) / (2*delta),
#   ∂v_z/∂z ≈ (v_z(i, j, k+1) - v_z(i, j, k-1)) / (2*delta).
# The boundary values of the divergence are set to zero since the derivatives are not well-defined
# at the boundaries using central differences.

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
    div = zeros((nx, ny, nz))

    # Compute the divergence for the interior points using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                div[i, j, k] = (
                    (v_x[i+1, j, k] - v_x[i-1, j, k]) / (2 * delta) +
                    (v_y[i, j+1, k] - v_y[i, j-1, k]) / (2 * delta) +
                    (v_z[i, j, k+1] - v_z[i, j, k-1]) / (2 * delta)
                )

    # The boundary values are already zero-initialized by zeros, so no need to set them explicitly.

    return div



# Background: The gradient of divergence operator combines two fundamental operations in vector calculus. 
# The divergence of a vector field measures the net flux of the field out of an infinitesimal volume, 
# while the gradient of a scalar field measures the rate and direction of change of the field. 
# Therefore, the gradient of the divergence of a vector field involves computing the divergence first 
# and then applying the gradient operation. However, doing these as separate steps can introduce errors 
# due to finite differencing. Instead, we derive a direct finite difference formula to compute this 
# operator more accurately for a 3D vector field on a discrete grid with spacing delta. 
# This approach ensures that the approximation remains second-order accurate for the interior points,
# while boundary values are zeroed out due to undefined derivatives there.

def grad_div(A_x, A_y, A_z, delta):
    '''Computes the gradient of the divergence of a 3D vector field using second-order finite differences.
    Parameters:
    -----------
    A_x : numpy.ndarray
        A 3D array representing the x-component of the vector field. Shape: (nx, ny, nz).
    A_y : numpy.ndarray
        A 3D array representing the y-component of the vector field. Shape: (nx, ny, nz).
    A_z : numpy.ndarray
        A 3D array representing the z-component of the vector field. Shape: (nx, ny, nz).
    delta : float
        The grid spacing or step size in all spatial directions.
    Returns:
    --------
    grad_div_x : numpy.ndarray
        A 3D array representing the x-component of the gradient of divergence. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    grad_div_y : numpy.ndarray
        A 3D array representing the y-component of the gradient of divergence. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    grad_div_z : numpy.ndarray
        A 3D array representing the z-component of the gradient of divergence. Shape: (nx, ny, nz).
        The boundary values are set to zero.
    '''


    nx, ny, nz = A_x.shape
    grad_div_x = zeros((nx, ny, nz))
    grad_div_y = zeros((nx, ny, nz))
    grad_div_z = zeros((nx, ny, nz))

    # Compute the gradient of divergence for the interior points using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Calculate the divergence's finite difference and directly compute its gradient
                d_div_x = (
                    (A_x[i+1, j, k] - 2*A_x[i, j, k] + A_x[i-1, j, k]) +
                    (A_y[i, j+1, k] - A_y[i, j-1, k]) +
                    (A_z[i, j, k+1] - A_z[i, j, k-1])
                ) / (delta**2)

                d_div_y = (
                    (A_x[i+1, j, k] - A_x[i-1, j, k]) +
                    (A_y[i, j+1, k] - 2*A_y[i, j, k] + A_y[i, j-1, k]) +
                    (A_z[i, j, k+1] - A_z[i, j, k-1])
                ) / (delta**2)

                d_div_z = (
                    (A_x[i+1, j, k] - A_x[i-1, j, k]) +
                    (A_y[i, j+1, k] - A_y[i, j-1, k]) +
                    (A_z[i, j, k+1] - 2*A_z[i, j, k] + A_z[i, j, k-1])
                ) / (delta**2)

                grad_div_x[i, j, k] = d_div_x
                grad_div_y[i, j, k] = d_div_y
                grad_div_z[i, j, k] = d_div_z

    # The boundary values are already zero-initialized by zeros, so no need to set them explicitly.

    return grad_div_x, grad_div_y, grad_div_z

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('13.5', 3)
target = targets[0]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
fx    = -y
fy    = x
fz    = np.zeros(z.shape)
delta = x[0][1][0] - x[0][0][0]
assert np.allclose(grad_div(fx, fy, fz, delta), target)
target = targets[1]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
fx    = x
fy    = y
fz    = z
delta = x[0][1][0] - x[0][0][0]
assert np.allclose(grad_div(fx, fy, fz, delta), target)
target = targets[2]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
fx    = np.sin(x*y*z)
fy    = np.cos(x*y*z)
fz    = fx*fy
delta = x[0][1][0] - x[0][0][0]
assert np.allclose(grad_div(fx, fy, fz, delta), target)
