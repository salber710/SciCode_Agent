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
    if delta <= 0:
        raise ValueError("delta must be a positive number")

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

def __init__(self, n_grid, x_out):
    """Constructor sets up coordinates, memory for variables.
        The variables:
            mesh points:
                x: the x coordinate for each mesh grid
                y: the y coordinate for each mesh grid
                z: the z coordinate for each mesh grid
                t: the time coordinate of the simulation
                r: the distance to the origin for each mesh grid
            evolving fields:
                E_x: the x component of the field E
                E_y: the y componnet of the field E
                E_z: the z component of the field E
                A_x: the x component of the field A
                A_y: the y component of the field A
                A_z: the z component of the field A
                phi: the scalar potential field phi values
            monitor variables:
                constraint: the current constraint violation value from the evolving fields.
                
        """
    self.n_grid = n_grid
    self.n_vars = 7
    self.delta = float(x_out) / (n_grid - 2.0)
    delta = self.delta
    self.x = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[:, None, None]
    self.y = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[None, :, None]
    self.z = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[None, None, :]
    self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    self.E_x = zeros((n_grid, n_grid, n_grid))
    self.E_y = zeros((n_grid, n_grid, n_grid))
    self.E_z = zeros((n_grid, n_grid, n_grid))
    self.A_x = zeros((n_grid, n_grid, n_grid))
    self.A_y = zeros((n_grid, n_grid, n_grid))
    self.A_z = zeros((n_grid, n_grid, n_grid))
    self.phi = zeros((n_grid, n_grid, n_grid))
    self.constraint = zeros((n_grid, n_grid, n_grid))
    self.t = 0.0


# Background: In computational physics, applying boundary conditions is crucial for ensuring that the numerical solution
# respects the physical constraints of the problem. For simulations involving symmetry, such as those in electromagnetism
# or fluid dynamics, boundary conditions can be applied to reflect the symmetry of the system. In a 3D grid, symmetry
# conditions can be applied at the boundaries to ensure that the field behaves correctly at the edges. For example, if a
# field is symmetric across a boundary, the field value at the boundary should be the same as the adjacent interior value.
# Conversely, if a field is antisymmetric, the field value at the boundary should be the negative of the adjacent interior
# value. This function applies such symmetry conditions to the time derivatives of a scalar field on a 3D grid.

def symmetry(f_dot, x_sym, y_sym, z_sym):
    '''Computes time derivatives on inner boundaries from symmetry
    Parameters:
    -----------
    f_dot : numpy.ndarray
        A 3D array representing the time derivatives of the scalar field. Shape: (nx, ny, nz).
        This array will be updated in-place with symmetric boundary conditions applied.
    x_sym : float
        The symmetry factor to apply along the x-axis (typically -1 for antisymmetry, 1 for symmetry).
    y_sym : float
        The symmetry factor to apply along the y-axis (typically -1 for antisymmetry, 1 for symmetry).
    z_sym : float
        The symmetry factor to apply along the z-axis (typically -1 for antisymmetry, 1 for symmetry).
    Returns:
    --------
    f_dot : numpy.ndarray
        The same 3D array passed in as input, with updated values at the boundaries according to the symmetry conditions.
        Shape: (nx, ny, nz).
    '''
    # Apply symmetry conditions at the x=0 boundary
    f_dot[0, :, :] = x_sym * f_dot[1, :, :]

    # Apply symmetry conditions at the y=0 boundary
    f_dot[:, 0, :] = y_sym * f_dot[:, 1, :]

    # Apply symmetry conditions at the z=0 boundary
    f_dot[:, :, 0] = z_sym * f_dot[:, :, 1]

    return f_dot

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('13.7', 3)
target = targets[0]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
f_dot = np.sin(x)*np.sin(2*y)*np.sin(3*z) 
assert np.allclose(symmetry(f_dot, 1, 1, -1), target)
target = targets[1]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
f_dot = (x+y)**2+z
assert np.allclose(symmetry(f_dot, -1, 1, 1), target)
target = targets[2]

x,y,z = np.meshgrid(*[np.linspace(-10,10,100)]*3)
f_dot = x**4 + y**2 + z**5
assert np.allclose(symmetry(f_dot, 1, -1, 1), target)
