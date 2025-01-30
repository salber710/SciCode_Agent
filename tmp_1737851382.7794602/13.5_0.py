from numpy import zeros, linspace, exp, sqrt
import numpy as np

# Background: In numerical analysis, the finite difference method is used to approximate derivatives of functions. 
# For a scalar field defined on a 3D grid, the partial derivatives with respect to each spatial direction can be 
# approximated using finite differences. A second-order central difference scheme provides a more accurate 
# approximation by considering the function values at points on either side of the point of interest. 
# For interior points, the second-order central difference for a function f with respect to x is given by:
# ∂f/∂x ≈ (f(x+Δx, y, z) - f(x-Δx, y, z)) / (2*Δx).
# At the boundaries, where central differences cannot be applied symmetrically, a second-order one-sided 
# difference is used. For example, at the lower boundary in the x-direction:
# ∂f/∂x ≈ (-3*f(x, y, z) + 4*f(x+Δx, y, z) - f(x+2Δx, y, z)) / (2*Δx).
# This approach is applied similarly for the y and z directions.


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

    # Compute partial derivatives using central differences for interior points
    deriv_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    deriv_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    deriv_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)

    # Compute partial derivatives using one-sided differences at the boundaries
    # x boundaries
    deriv_x[0, :, :] = (-3 * fct[0, :, :] + 4 * fct[1, :, :] - fct[2, :, :]) / (2 * delta)
    deriv_x[-1, :, :] = (3 * fct[-1, :, :] - 4 * fct[-2, :, :] + fct[-3, :, :]) / (2 * delta)

    # y boundaries
    deriv_y[:, 0, :] = (-3 * fct[:, 0, :] + 4 * fct[:, 1, :] - fct[:, 2, :]) / (2 * delta)
    deriv_y[:, -1, :] = (3 * fct[:, -1, :] - 4 * fct[:, -2, :] + fct[:, -3, :]) / (2 * delta)

    # z boundaries
    deriv_z[:, :, 0] = (-3 * fct[:, :, 0] + 4 * fct[:, :, 1] - fct[:, :, 2]) / (2 * delta)
    deriv_z[:, :, -1] = (3 * fct[:, :, -1] - 4 * fct[:, :, -2] + fct[:, :, -3]) / (2 * delta)

    return deriv_x, deriv_y, deriv_z


# Background: The Laplacian operator, denoted as ∇², is a second-order differential operator in n-dimensional Euclidean space, 
# defined as the divergence of the gradient of a function. In the context of a scalar field on a 3D grid, the Laplacian 
# measures the rate at which the average value of the field around a point differs from the value at that point. 
# For a function f(x, y, z), the Laplacian is given by:
# ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².
# Using a second-order central finite difference scheme, the second derivatives can be approximated as:
# ∂²f/∂x² ≈ (f(x+Δx, y, z) - 2*f(x, y, z) + f(x-Δx, y, z)) / (Δx²).
# Similar expressions apply for the y and z directions. The Laplacian is computed for interior points, and boundary values 
# are set to zero to ensure the calculation is confined to the interior of the grid.


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
                lap[i, j, k] = (
                    (fct[i+1, j, k] - 2 * fct[i, j, k] + fct[i-1, j, k]) +
                    (fct[i, j+1, k] - 2 * fct[i, j, k] + fct[i, j-1, k]) +
                    (fct[i, j, k+1] - 2 * fct[i, j, k] + fct[i, j, k-1])
                ) / (delta ** 2)

    # Boundary values are already set to zero by the initialization of lap with zeros
    return lap



# Background: The gradient of a scalar field is a vector field that points in the direction of the greatest rate of increase of the scalar field. 
# In a 3D grid, the gradient is composed of the partial derivatives of the scalar field with respect to each spatial direction. 
# For a scalar field f(x, y, z), the gradient is given by the vector (∂f/∂x, ∂f/∂y, ∂f/∂z). 
# Using a second-order central finite difference scheme, the partial derivatives can be approximated for interior points as:
# ∂f/∂x ≈ (f(x+Δx, y, z) - f(x-Δx, y, z)) / (2*Δx),
# ∂f/∂y ≈ (f(x, y+Δy, z) - f(x, y-Δy, z)) / (2*Δy),
# ∂f/∂z ≈ (f(x, y, z+Δz) - f(x, y, z-Δz)) / (2*Δz).
# The boundary values are set to zero to ensure the gradient is only calculated for the interior grid points.

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

    # Compute the gradient using central differences for interior points
    grad_x[1:-1, :, :] = (fct[2:, :, :] - fct[:-2, :, :]) / (2 * delta)
    grad_y[:, 1:-1, :] = (fct[:, 2:, :] - fct[:, :-2, :]) / (2 * delta)
    grad_z[:, :, 1:-1] = (fct[:, :, 2:] - fct[:, :, :-2]) / (2 * delta)

    # Boundary values are already set to zero by the initialization of grad_x, grad_y, grad_z with zeros
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

    # Compute the divergence using central differences for interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                div[i, j, k] = (
                    (v_x[i+1, j, k] - v_x[i-1, j, k]) / (2 * delta) +
                    (v_y[i, j+1, k] - v_y[i, j-1, k]) / (2 * delta) +
                    (v_z[i, j, k+1] - v_z[i, j, k-1]) / (2 * delta)
                )

    # Boundary values are already set to zero by the initialization of div with zeros
    return div



# Background: The gradient of the divergence of a vector field is a higher-order differential operator that combines
# the concepts of divergence and gradient. The divergence of a vector field measures the net rate of flow out of a point,
# while the gradient of a scalar field points in the direction of the greatest rate of increase of the field.
# To compute the gradient of the divergence directly using finite differences, we need to derive a formula that
# approximates this operation without first computing the divergence separately, as doing so would introduce larger errors.
# The second-order finite difference scheme is used to approximate derivatives, ensuring accuracy for interior points
# while setting boundary values to zero.

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
    grad_div_x = np.zeros_like(A_x)
    grad_div_y = np.zeros_like(A_y)
    grad_div_z = np.zeros_like(A_z)

    # Compute the gradient of the divergence using central differences for interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Compute the divergence at the current point
                div = (
                    (A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) +
                    (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) +
                    (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta)
                )
                
                # Compute the gradient of the divergence
                grad_div_x[i, j, k] = (div - (A_x[i-1, j, k] - 2 * A_x[i, j, k] + A_x[i+1, j, k])) / (delta ** 2)
                grad_div_y[i, j, k] = (div - (A_y[i, j-1, k] - 2 * A_y[i, j, k] + A_y[i, j+1, k])) / (delta ** 2)
                grad_div_z[i, j, k] = (div - (A_z[i, j, k-1] - 2 * A_z[i, j, k] + A_z[i, j, k+1])) / (delta ** 2)

    # Boundary values are already set to zero by the initialization of grad_div_x, grad_div_y, grad_div_z with zeros
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
