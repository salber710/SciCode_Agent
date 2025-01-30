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

    # Precompute the reciprocal of delta for greater efficiency
    inv_delta = 1.0 / (2.0 * delta)

    # Calculate divergence using a different pattern: use a staggered grid approach
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Use an alternating pattern for computation
                if (i + j + k) % 2 == 0:
                    div[i, j, k] = (
                        ((v_x[i+1, j, k] - v_x[i-1, j, k]) +
                         (v_y[i, j+1, k] - v_y[i, j-1, k]) +
                         (v_z[i, j, k+1] - v_z[i, j, k-1])) * inv_delta
                    )
                else:
                    # Apply alternative approach for staggered points
                    div[i, j, k] = ((v_x[i+1, j, k] - v_x[i, j, k]) / delta +
                                    (v_y[i, j+1, k] - v_y[i, j, k]) / delta +
                                    (v_z[i, j, k+1] - v_z[i, j, k]) / delta)

    # Boundary values are zero by initialization
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

    # Get the shape of the input field components
    nx, ny, nz = A_x.shape

    # Initialize the gradient of divergence arrays with zeros
    grad_div_x = np.zeros((nx, ny, nz), dtype=A_x.dtype)
    grad_div_y = np.zeros((nx, ny, nz), dtype=A_y.dtype)
    grad_div_z = np.zeros((nx, ny, nz), dtype=A_z.dtype)

    # Precompute factor for efficiency
    inv_delta_sq = 1.0 / (delta * delta)

    # Compute second-order finite differences for gradient of divergence
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                # Compute divergence using a mixed centered and staggered difference
                div = (
                    (A_x[i+1, j, k] - A_x[i, j, k]) * 0.5 +
                    (A_x[i, j, k] - A_x[i-1, j, k]) * 0.5 +
                    (A_y[i, j+1, k] - A_y[i, j, k]) * 0.5 +
                    (A_y[i, j, k] - A_y[i, j-1, k]) * 0.5 +
                    (A_z[i, j, k+1] - A_z[i, j, k]) * 0.5 +
                    (A_z[i, j, k] - A_z[i, j, k-1]) * 0.5
                ) / delta

                # Use a different pattern for gradient calculation
                grad_div_x[i, j, k] = (
                    (div + (A_x[i+1, j, k] - A_x[i-1, j, k])) * inv_delta_sq
                )
                grad_div_y[i, j, k] = (
                    (div + (A_y[i, j+1, k] - A_y[i, j-1, k])) * inv_delta_sq
                )
                grad_div_z[i, j, k] = (
                    (div + (A_z[i, j, k+1] - A_z[i, j, k-1])) * inv_delta_sq
                )

    # Boundary values are zero by initialization
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

def apply_symmetry_conditions_hybrid(f_dot, x_sym, y_sym, z_sym):
    '''Applies symmetry or antisymmetry conditions on the inner boundaries using a hybrid approach of slicing and iteration.
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
    None
    '''
    
    nx, ny, nz = f_dot.shape

    # Mixed slicing and iteration for x-symmetry
    f_dot[0, :, :] = x_sym * f_dot[1, :, :]
    for j in range(1, ny, 2):
        f_dot[0, j, :] = x_sym * f_dot[1, j, :]

    # Mixed slicing and iteration for y-symmetry
    f_dot[:, 0, :] = y_sym * f_dot[:, 1, :]
    for i in range(1, nx, 2):
        f_dot[i, 0, :] = y_sym * f_dot[i, 1, :]

    # Mixed slicing and iteration for z-symmetry
    f_dot[:, :, 0] = z_sym * f_dot[:, :, 1]
    for i in range(1, nx, 2):
        for j in range(1, ny, 2):
            f_dot[i, j, 0] = z_sym * f_dot[i, j, 1]


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


    # Retrieve necessary parameters from the Maxwell object
    delta = maxwell.delta
    r = maxwell.r

    # Get the shape of the grid
    nx, ny, nz = f.shape

    # Implement a boundary condition using a hyperbolic tangent as a smooth transition factor
    # This creates a gradual transition to zero at the boundary, reducing reflections
    tanh_factor = np.tanh(r / delta)

    # Apply the outgoing wave condition with this hyperbolic tangent factor
    # x-boundary
    f_dot[-1, :, :] = tanh_factor[-1, :, :] * (f[-1, :, :] - f[-2, :, :]) / (r[-1, :, :] * delta)

    # y-boundary
    f_dot[:, -1, :] = tanh_factor[:, -1, :] * (f[:, -1, :] - f[:, -2, :]) / (r[:, -1, :] * delta)

    # z-boundary
    f_dot[:, :, -1] = tanh_factor[:, :, -1] * (f[:, :, -1] - f[:, :, -2]) / (r[:, :, -1] * delta)

    return f_dot




def derivatives(maxwell, fields):
    '''Computes the time derivatives of electromagnetic fields according to Maxwell's equations in Lorentz Gauge.
    Parameters:
    -----------
    maxwell : object
        An object containing properties of the simulation grid and field values, including:
        - `A_x`, `A_y`, `A_z`: 3D arrays representing the vector potential components.
        - `E_x`, `E_y`, `E_z`: 3D arrays representing the electric field components.
        - `phi`: 3D array representing the scalar potential.
        - `delta`: Grid spacing (step size) in all spatial directions.
    fields : tuple of numpy.ndarray
        A tuple containing the current field values in the following order:
        `(E_x, E_y, E_z, A_x, A_y, A_z, phi)`.
        Each component is a 3D array of shape `(nx, ny, nz)`.
    Returns:
    --------
    tuple of numpy.ndarray
        A tuple containing the time derivatives of the fields in the following order:
        `(E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)`.
        Each component is a 3D array of shape `(nx, ny, nz)`.
    '''
    E_x, E_y, E_z, A_x, A_y, A_z, phi = fields
    delta = maxwell.delta

    # Function to compute the Laplacian
    def laplacian(f):
        return (
            (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / delta**2 +
            (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / delta**2 +
            (np.roll(f, -1, axis=2) - 2*f + np.roll(f, 1, axis=2)) / delta**2
        )

    # Function to compute the divergence of A
    def divergence(A_x, A_y, A_z):
        div_x = (np.roll(A_x, -1, axis=0) - np.roll(A_x, 1, axis=0)) / (2 * delta)
        div_y = (np.roll(A_y, -1, axis=1) - np.roll(A_y, 1, axis=1)) / (2 * delta)
        div_z = (np.roll(A_z, -1, axis=2) - np.roll(A_z, 1, axis=2)) / (2 * delta)
        return div_x + div_y + div_z

    # Function to apply boundary conditions
    def apply_boundary_conditions(field):
        # Mirror symmetry at z = 0
        field[:, :, 0] = field[:, :, 1]
        # Cylindrical symmetry at x = 0
        field[0, :, :] = field[1, :, :]
        # Cylindrical symmetry at y = 0
        field[:, 0, :] = field[:, 1, :]
        # Outgoing wave condition on the outer boundary
        field[-1, :, :] = field[-2, :, :]
        field[:, -1, :] = field[:, -2, :]
        field[:, :, -1] = field[:, :, -2]

    # Compute time derivatives
    E_x_dot = -laplacian(A_x) + divergence(A_x, A_y, A_z)
    E_y_dot = -laplacian(A_y) + divergence(A_x, A_y, A_z)
    E_z_dot = -laplacian(A_z) + divergence(A_x, A_y, A_z)

    grad_phi_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * delta)
    grad_phi_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * delta)
    grad_phi_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * delta)

    A_x_dot = -E_x - grad_phi_x
    A_y_dot = -E_y - grad_phi_y
    A_z_dot = -E_z - grad_phi_z

    phi_dot = -divergence(A_x, A_y, A_z)

    # Apply boundary conditions
    apply_boundary_conditions(E_x_dot)
    apply_boundary_conditions(E_y_dot)
    apply_boundary_conditions(E_z_dot)
    apply_boundary_conditions(A_x_dot)
    apply_boundary_conditions(A_y_dot)
    apply_boundary_conditions(A_z_dot)
    apply_boundary_conditions(phi_dot)

    return E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot


try:
    targets = process_hdf5_to_tuple('13.9', 3)
    target = targets[0]
    maxwell = Maxwell(50,2)
    x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
    fields = (x,y,z,x,y,z, z*0)
    assert np.allclose(derivatives(maxwell, fields), target)

    target = targets[1]
    maxwell = Maxwell(50,2)
    x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
    fields = (x,y,z,-y,x,z*0, z*0+1)
    assert np.allclose(derivatives(maxwell, fields), target)

    target = targets[2]
    maxwell = Maxwell(50,2)
    x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
    fields = (x,y,z,x*0,-z,y, z*0+1)
    assert np.allclose(derivatives(maxwell, fields), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e