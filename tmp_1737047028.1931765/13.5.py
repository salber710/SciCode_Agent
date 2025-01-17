from numpy import zeros, linspace, exp, sqrt
import numpy as np

# Background: In numerical analysis, the finite difference method is used to approximate derivatives of functions. 
# For a scalar field defined on a 3D grid, the partial derivatives with respect to each spatial direction can be 
# approximated using finite differences. A second-order central difference scheme provides a good balance between 
# accuracy and computational efficiency. For interior points, the second-order central difference is used, while 
# for boundary points, a second-order one-sided difference is applied to maintain accuracy without requiring 
# values outside the domain. The central difference for a function f at a point i is given by:
# (f[i+1] - f[i-1]) / (2*delta), where delta is the grid spacing. For boundary points, the forward or backward 
# difference is used: (f[i+2] - 4*f[i+1] + 3*f[i]) / (2*delta) for the start, and (-3*f[i] + 4*f[i-1] - f[i-2]) / (2*delta) 
# for the end.

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

    # Compute partial derivatives with respect to x
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if 1 <= i < nx - 1:
                    deriv_x[i, j, k] = (fct[i + 1, j, k] - fct[i - 1, j, k]) / (2 * delta)
                elif i == 0:
                    deriv_x[i, j, k] = (-3 * fct[i, j, k] + 4 * fct[i + 1, j, k] - fct[i + 2, j, k]) / (2 * delta)
                elif i == nx - 1:
                    deriv_x[i, j, k] = (3 * fct[i, j, k] - 4 * fct[i - 1, j, k] + fct[i - 2, j, k]) / (2 * delta)

    # Compute partial derivatives with respect to y
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if 1 <= j < ny - 1:
                    deriv_y[i, j, k] = (fct[i, j + 1, k] - fct[i, j - 1, k]) / (2 * delta)
                elif j == 0:
                    deriv_y[i, j, k] = (-3 * fct[i, j, k] + 4 * fct[i, j + 1, k] - fct[i, j + 2, k]) / (2 * delta)
                elif j == ny - 1:
                    deriv_y[i, j, k] = (3 * fct[i, j, k] - 4 * fct[i, j - 1, k] + fct[i, j - 2, k]) / (2 * delta)

    # Compute partial derivatives with respect to z
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if 1 <= k < nz - 1:
                    deriv_z[i, j, k] = (fct[i, j, k + 1] - fct[i, j, k - 1]) / (2 * delta)
                elif k == 0:
                    deriv_z[i, j, k] = (-3 * fct[i, j, k] + 4 * fct[i, j, k + 1] - fct[i, j, k + 2]) / (2 * delta)
                elif k == nz - 1:
                    deriv_z[i, j, k] = (3 * fct[i, j, k] - 4 * fct[i, j, k - 1] + fct[i, j, k - 2]) / (2 * delta)

    return deriv_x, deriv_y, deriv_z


# Background: The Laplacian operator, denoted as ∇², is a second-order differential operator in n-dimensional Euclidean space, 
# defined as the divergence of the gradient of a function. In the context of a scalar field on a 3D grid, the Laplacian 
# measures the rate at which the average value of the field around a point differs from the value at that point. 
# It is widely used in physics and engineering, particularly in the study of heat conduction, fluid dynamics, and 
# electromagnetism. For a discrete grid, the Laplacian can be approximated using finite differences. 
# The second-order central difference scheme is used for interior points, which provides a balance between accuracy 
# and computational efficiency. The Laplacian in 3D is given by the sum of second partial derivatives with respect 
# to each spatial direction: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z². For boundary points, the Laplacian is set to zero 
# to ensure calculations are only performed on interior points.

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

    # Compute Laplacian for interior points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                lap[i, j, k] = (
                    (fct[i + 1, j, k] - 2 * fct[i, j, k] + fct[i - 1, j, k]) / (delta ** 2) +
                    (fct[i, j + 1, k] - 2 * fct[i, j, k] + fct[i, j - 1, k]) / (delta ** 2) +
                    (fct[i, j, k + 1] - 2 * fct[i, j, k] + fct[i, j, k - 1]) / (delta ** 2)
                )

    # Boundary values are already set to zero by np.zeros_like
    return lap


# Background: The gradient of a scalar field is a vector field that points in the direction of the greatest rate of increase of the scalar field. 
# In three dimensions, the gradient is composed of the partial derivatives of the scalar field with respect to each spatial direction: 
# ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z). For a discrete grid, these partial derivatives can be approximated using finite differences. 
# A second-order central difference scheme is used for interior points to achieve a balance between accuracy and computational efficiency. 
# For boundary points, the gradient is set to zero to ensure calculations are only performed on interior points.

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

    # Compute gradient for interior points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                grad_x[i, j, k] = (fct[i + 1, j, k] - fct[i - 1, j, k]) / (2 * delta)
                grad_y[i, j, k] = (fct[i, j + 1, k] - fct[i, j - 1, k]) / (2 * delta)
                grad_z[i, j, k] = (fct[i, j, k + 1] - fct[i, j, k - 1]) / (2 * delta)

    # Boundary values are already set to zero by np.zeros_like
    return grad_x, grad_y, grad_z


# Background: The divergence of a vector field is a scalar field that represents the rate at which "density" exits a point in space. 
# In the context of a 3D vector field, the divergence is calculated as the sum of the partial derivatives of each component of the vector field 
# with respect to its corresponding spatial direction: ∇·v = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z. For a discrete grid, these partial derivatives can be 
# approximated using finite differences. A second-order central difference scheme is used for interior points to achieve a balance between 
# accuracy and computational efficiency. For boundary points, the divergence is set to zero to ensure calculations are only performed on 
# interior points.

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

    # Compute divergence for interior points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                div[i, j, k] = (
                    (v_x[i + 1, j, k] - v_x[i - 1, j, k]) / (2 * delta) +
                    (v_y[i, j + 1, k] - v_y[i, j - 1, k]) / (2 * delta) +
                    (v_z[i, j, k + 1] - v_z[i, j, k - 1]) / (2 * delta)
                )

    # Boundary values are already set to zero by np.zeros_like
    return div



# Background: The gradient of the divergence of a vector field is a higher-order differential operator that combines
# the concepts of divergence and gradient. The divergence of a vector field gives a scalar field representing the
# net rate of flow out of a point, while the gradient of a scalar field gives a vector field pointing in the direction
# of the greatest rate of increase. To compute the gradient of the divergence directly using finite differences,
# we need to apply the finite difference approximation to the second derivatives involved in this operation.
# This involves calculating mixed second partial derivatives of the vector field components, which can be done
# using a second-order central difference scheme for interior points. Boundary values are set to zero to ensure
# calculations are only performed on interior points.

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

    # Compute gradient of divergence for interior points
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                # Compute second derivatives for the gradient of divergence
                grad_div_x[i, j, k] = (
                    (A_x[i + 1, j, k] - 2 * A_x[i, j, k] + A_x[i - 1, j, k]) / (delta ** 2) +
                    (A_y[i, j + 1, k] - A_y[i, j - 1, k]) / (2 * delta) +
                    (A_z[i, j, k + 1] - A_z[i, j, k - 1]) / (2 * delta)
                )
                grad_div_y[i, j, k] = (
                    (A_x[i + 1, j, k] - A_x[i - 1, j, k]) / (2 * delta) +
                    (A_y[i, j + 1, k] - 2 * A_y[i, j, k] + A_y[i, j - 1, k]) / (delta ** 2) +
                    (A_z[i, j, k + 1] - A_z[i, j, k - 1]) / (2 * delta)
                )
                grad_div_z[i, j, k] = (
                    (A_x[i + 1, j, k] - A_x[i - 1, j, k]) / (2 * delta) +
                    (A_y[i, j + 1, k] - A_y[i, j - 1, k]) / (2 * delta) +
                    (A_z[i, j, k + 1] - 2 * A_z[i, j, k] + A_z[i, j, k - 1]) / (delta ** 2)
                )

    # Boundary values are already set to zero by np.zeros_like
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
