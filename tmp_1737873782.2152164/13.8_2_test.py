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
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                div[i, j, k] = (v_x[i+1, j, k] - v_x[i-1, j, k]) / (2 * delta) + \
                               (v_y[i, j+1, k] - v_y[i, j-1, k]) / (2 * delta) + \
                               (v_z[i, j, k+1] - v_z[i, j, k-1]) / (2 * delta)
    
    # Set boundary values to zero (already initialized to zero)
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

    # Initialize the gradient of divergence arrays
    nx, ny, nz = A_x.shape
    grad_div_x = np.zeros((nx, ny, nz))
    grad_div_y = np.zeros((nx, ny, nz))
    grad_div_z = np.zeros((nx, ny, nz))

    # Compute the gradient of divergence using central differences for interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Compute divergence and then its gradient
                div = (
                    (A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) +
                    (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) +
                    (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta)
                )
                grad_div_x[i, j, k] = (div - (A_x[i-1, j, k] - 2 * A_x[i, j, k] + A_x[i+1, j, k]) / (delta**2)) / (2 * delta)
                grad_div_y[i, j, k] = (div - (A_y[i, j-1, k] - 2 * A_y[i, j, k] + A_y[i, j+1, k]) / (delta**2)) / (2 * delta)
                grad_div_z[i, j, k] = (div - (A_z[i, j, k-1] - 2 * A_z[i, j, k] + A_z[i, j, k+1]) / (delta**2)) / (2 * delta)

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

    # Apply boundary conditions for x = 0
    f_dot[0, :, :] = x_sym * f_dot[1, :, :]
    
    # Apply boundary conditions for y = 0
    f_dot[:, 0, :] = y_sym * f_dot[:, 1, :]
    
    # Apply boundary conditions for z = 0
    f_dot[:, :, 0] = z_sym * f_dot[:, :, 1]

    return f_dot



def outgoing_wave(maxwell, f_dot, f):
    '''Computes time derivatives of fields from outgoing-wave boundary condition
    Parameters:
    -----------
    maxwell : object
        An object containing properties of the simulation grid, including:
        - `delta`: Grid spacing (step size) in all spatial directions.
        - `x`, `y`, `z`: 3D arrays representing the coordinate grids along the x, y, and z axes, respectively.
        - `r`: 3D array representing the grid radial distance from the origin.
    f_dot : numpy.ndarray
        A 3D array representing the time derivatives of the scalar field. Shape: (nx, ny, nz).
        This array will be updated in-place with the outgoing wave boundary condition applied.
    f : numpy.ndarray
        A 3D array representing the scalar field values on the grid. Shape: (nx, ny, nz).
    Returns:
    --------
    f_dot : numpy.ndarray
        The same 3D array passed in as input, with updated values at the outer boundaries according to the
        outgoing wave boundary condition. Shape: (nx, ny, nz).
    '''

    # Get grid dimensions and delta
    nx, ny, nz = f.shape
    delta = maxwell.delta

    # Apply outgoing wave boundary condition
    # For outgoing wave boundary, the time derivative at the boundary is adjusted based on the spatial derivative
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Check if we're at the outer boundary
                if i == nx - 1 or j == ny - 1 or k == nz - 1:
                    # Compute distance to boundary
                    r_boundary = sqrt(maxwell.x[i, j, k]**2 + maxwell.y[i, j, k]**2 + maxwell.z[i, j, k]**2)
                    
                    # Calculate the outgoing wave condition
                    if r_boundary != 0:
                        f_dot[i, j, k] = -f[i, j, k] / r_boundary
                    else:
                        f_dot[i, j, k] = 0

    return f_dot


try:
    targets = process_hdf5_to_tuple('13.8', 3)
    target = targets[0]
    maxwell = Maxwell(50,2)
    x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
    f_dot = np.sin(x)*np.sin(2*y)*np.sin(3*z) 
    f     = 2*f_dot
    assert np.allclose(outgoing_wave(maxwell, f_dot, f), target)

    target = targets[1]
    maxwell = Maxwell(50,2)
    x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
    f_dot = (x+y)**2+z 
    f     = 2*f_dot
    assert np.allclose(outgoing_wave(maxwell, f_dot, f), target)

    target = targets[2]
    maxwell = Maxwell(50,2)
    x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
    f_dot = x**4 + y**2 + z**5
    f     = 2*f_dot
    assert np.allclose(outgoing_wave(maxwell, f_dot, f), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e