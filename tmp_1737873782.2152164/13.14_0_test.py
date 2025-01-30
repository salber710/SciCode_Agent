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
    
    # Get the shape of the field
    nx, ny, nz = f.shape
    delta = maxwell.delta

    # Apply outgoing wave boundary condition at the outer boundaries
    # For x = max boundary
    f_dot[-1, :, :] = f_dot[-2, :, :] - (f[-1, :, :] - f[-2, :, :]) / delta
    
    # For y = max boundary
    f_dot[:, -1, :] = f_dot[:, -2, :] - (f[:, -1, :] - f[:, -2, :]) / delta
    
    # For z = max boundary
    f_dot[:, :, -1] = f_dot[:, :, -2] - (f[:, :, -1] - f[:, :, -2]) / delta

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

    # Initialize the derivative arrays
    E_x_dot = np.zeros_like(E_x)
    E_y_dot = np.zeros_like(E_y)
    E_z_dot = np.zeros_like(E_z)
    A_x_dot = np.zeros_like(A_x)
    A_y_dot = np.zeros_like(A_y)
    A_z_dot = np.zeros_like(A_z)
    phi_dot = np.zeros_like(phi)

    # Compute the divergence of A and the Laplacian of A
    div_A = (
        (A_x[2:, 1:-1, 1:-1] - A_x[:-2, 1:-1, 1:-1]) / (2 * delta) +
        (A_y[1:-1, 2:, 1:-1] - A_y[1:-1, :-2, 1:-1]) / (2 * delta) +
        (A_z[1:-1, 1:-1, 2:] - A_z[1:-1, 1:-1, :-2]) / (2 * delta)
    )

    lap_A_x = (
        (A_x[2:, 1:-1, 1:-1] - 2 * A_x[1:-1, 1:-1, 1:-1] + A_x[:-2, 1:-1, 1:-1]) / (delta**2) +
        (A_x[1:-1, 2:, 1:-1] - 2 * A_x[1:-1, 1:-1, 1:-1] + A_x[1:-1, :-2, 1:-1]) / (delta**2) +
        (A_x[1:-1, 1:-1, 2:] - 2 * A_x[1:-1, 1:-1, 1:-1] + A_x[1:-1, 1:-1, :-2]) / (delta**2)
    )

    lap_A_y = (
        (A_y[2:, 1:-1, 1:-1] - 2 * A_y[1:-1, 1:-1, 1:-1] + A_y[:-2, 1:-1, 1:-1]) / (delta**2) +
        (A_y[1:-1, 2:, 1:-1] - 2 * A_y[1:-1, 1:-1, 1:-1] + A_y[1:-1, :-2, 1:-1]) / (delta**2) +
        (A_y[1:-1, 1:-1, 2:] - 2 * A_y[1:-1, 1:-1, 1:-1] + A_y[1:-1, 1:-1, :-2]) / (delta**2)
    )

    lap_A_z = (
        (A_z[2:, 1:-1, 1:-1] - 2 * A_z[1:-1, 1:-1, 1:-1] + A_z[:-2, 1:-1, 1:-1]) / (delta**2) +
        (A_z[1:-1, 2:, 1:-1] - 2 * A_z[1:-1, 1:-1, 1:-1] + A_z[1:-1, :-2, 1:-1]) / (delta**2) +
        (A_z[1:-1, 1:-1, 2:] - 2 * A_z[1:-1, 1:-1, 1:-1] + A_z[1:-1, 1:-1, :-2]) / (delta**2)
    )

    # Compute the time derivatives
    E_x_dot[1:-1, 1:-1, 1:-1] = -lap_A_x + (div_A - (A_x[2:, 1:-1, 1:-1] - A_x[:-2, 1:-1, 1:-1]) / (2 * delta))
    E_y_dot[1:-1, 1:-1, 1:-1] = -lap_A_y + (div_A - (A_y[1:-1, 2:, 1:-1] - A_y[1:-1, :-2, 1:-1]) / (2 * delta))
    E_z_dot[1:-1, 1:-1, 1:-1] = -lap_A_z + (div_A - (A_z[1:-1, 1:-1, 2:] - A_z[1:-1, 1:-1, :-2]) / (2 * delta))

    A_x_dot[1:-1, 1:-1, 1:-1] = -E_x[1:-1, 1:-1, 1:-1] - (phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2 * delta)
    A_y_dot[1:-1, 1:-1, 1:-1] = -E_y[1:-1, 1:-1, 1:-1] - (phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2 * delta)
    A_z_dot[1:-1, 1:-1, 1:-1] = -E_z[1:-1, 1:-1, 1:-1] - (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) / (2 * delta)

    phi_dot[1:-1, 1:-1, 1:-1] = -div_A

    # Apply boundary conditions
    symmetry(E_x_dot, -1, 1, 1)
    symmetry(E_y_dot, 1, -1, 1)
    symmetry(E_z_dot, 1, 1, -1)

    outgoing_wave(maxwell, E_x_dot, E_x)
    outgoing_wave(maxwell, E_y_dot, E_y)
    outgoing_wave(maxwell, E_z_dot, E_z)

    outgoing_wave(maxwell, A_x_dot, A_x)
    outgoing_wave(maxwell, A_y_dot, A_y)
    outgoing_wave(maxwell, A_z_dot, A_z)

    outgoing_wave(maxwell, phi_dot, phi)

    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


def update_fields(maxwell, fields, fields_dot, factor, dt):
    '''Updates all fields by adding a scaled increment of the time derivatives.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, including:
        - `n_vars`: Number of variables in the simulation (e.g., number of field components).
    fields : list of numpy.ndarray
        A list containing the current field values to be updated, in the following order:
        `[E_x, E_y, E_z, A_x, A_y, A_z, phi]`.
        Each field is a 3D array of shape `(nx, ny, nz)`.
    fields_dot : list of numpy.ndarray
        A list containing the time derivatives of the corresponding fields, in the same order as `fields`.
        Each derivative is a 3D array of shape `(nx, ny, nz)`.
    factor : float
        A scaling factor to be applied to the field updates. (useful when applying in the higher order time integrator like Runge-Kutta)
    dt : float
        Time step size.
    Returns:
    --------
    list of numpy.ndarray
        A list containing the updated fields in the same order as `fields`.
        Each updated field is a 3D array of shape `(nx, ny, nz)`.
    '''
    
    # Initialize a list to store the updated fields
    new_fields = []

    # Iterate over each field and its corresponding derivative
    for field, field_dot in zip(fields, fields_dot):
        # Compute the updated field by adding the scaled time derivative
        updated_field = field + factor * field_dot * dt
        new_fields.append(updated_field)

    return new_fields


def stepper(maxwell, fields, courant, t_const):
    '''Executes an iterative Crank-Nicholson (ICN) step using a Runge-Kutta scheme 
    to integrate from the current time to `t + t_const`.
    The ICN function uses a second-order scheme equivalent to the iterative Crank-Nicholson algorithm.
    The substep size `delta_t` is controlled by the Courant number to ensure stability.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, which includes:
        - `delta`: The grid spacing (common for all dimensions).
        - `t`: The current time of the simulation.
    fields : list of numpy.ndarray
        A list containing the current field values to be updated, in the following order:
        `[E_x, E_y, E_z, A_x, A_y, A_z, phi]`.
        Each field is a 3D array of shape `(nx, ny, nz)`.
    courant : float
        Courant number to control the substep size `delta_t` for stability.
    t_const : float
        The total time increment over which the simulation should be integrated.
    Returns:
    --------
    tuple (float, list of numpy.ndarray)
        Returns the updated simulation time and the updated fields after the integration.
        The updated fields are in the same order and shapes as `fields`.
    '''
    
    # Determine the substep size using the Courant number
    delta_t = courant * maxwell.delta

    # Calculate the number of substeps required to cover the total time increment
    num_steps = int(np.ceil(t_const / delta_t))
    
    # Adjust delta_t to exactly cover t_const with the given number of steps
    delta_t = t_const / num_steps

    # Initialize the current time
    current_time = maxwell.t

    # Iteratively apply the Crank-Nicholson method
    for _ in range(num_steps):
        # Calculate time derivatives at the current state
        fields_dot = derivatives(maxwell, fields)
        
        # First intermediate step (predictor step)
        intermediate_fields = update_fields(maxwell, fields, fields_dot, 1.0, delta_t / 2)

        # Calculate time derivatives at the intermediate state
        fields_dot_intermediate = derivatives(maxwell, intermediate_fields)
        
        # Second intermediate step (corrector step)
        fields = update_fields(maxwell, fields, fields_dot_intermediate, 1.0, delta_t)

        # Update the current time
        current_time += delta_t
    
    # Update the simulation time in the Maxwell object
    maxwell.t = current_time

    return current_time, fields


def check_constraint(maxwell):
    '''Check the constraint violation for the electric field components in the Maxwell object.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, containing the following attributes:
        - `E_x`, `E_y`, `E_z`: 3D arrays representing the components of the electric field.
        - `delta`: The grid spacing (assumed to be uniform for all dimensions).
        - `t`: The current time of the simulation.
    Returns:
    --------
    float
        The L2 norm of the constraint violation, calculated as the square root of the sum 
        of squares of the divergence values scaled by the grid cell volume.
    '''

    # Extract the electric field components and grid spacing
    E_x = maxwell.E_x
    E_y = maxwell.E_y
    E_z = maxwell.E_z
    delta = maxwell.delta
    
    # Compute the divergence of the electric field
    div_E = (
        (E_x[2:, 1:-1, 1:-1] - E_x[:-2, 1:-1, 1:-1]) / (2 * delta) +
        (E_y[1:-1, 2:, 1:-1] - E_y[1:-1, :-2, 1:-1]) / (2 * delta) +
        (E_z[1:-1, 1:-1, 2:] - E_z[1:-1, 1:-1, :-2]) / (2 * delta)
    )
    
    # Calculate the L2 norm of the constraint violation
    norm_c = np.sqrt(np.sum(div_E**2) * delta**3)

    return norm_c


def integrate(maxwell, courant, t_max, t_check):
    '''Carry out time integration.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, containing the following attributes:
        - `E_x`, `E_y`, `E_z`: 3D arrays representing the components of the electric field.
        - `A_x`, `A_y`, `A_z`: 3D arrays representing the components of the vector potential.
        - 'phi'  : the scalar potential
        - `delta`: The grid spacing (assumed to be uniform for all dimensions).
        - `t`: The current time of the simulation.
    courant: float
        the courant stability factor on dt
    t_max  : float
        the integration time upper limit
    t_check: float
        the simulation time duration to monitor the constraint violation. 
        Basically, every t_check simulation time unit, the constraint violation is recorded
    Returns:
    --------
    constraints: np.array
        the array containing the constraint violation on each time grid spaced by t_check. 
    '''
    
    # Initialize a list to record constraint violations
    constraints = []

    # Extract the initial fields
    fields = [maxwell.E_x, maxwell.E_y, maxwell.E_z, maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi]

    # Initialize the current time
    current_time = maxwell.t

    # Run the integration loop until the maximum time is reached
    while current_time < t_max:
        # Determine the next check time
        next_check_time = min(current_time + t_check, t_max)

        # Integrate up to the next check time
        current_time, fields = stepper(maxwell, fields, courant, next_check_time - current_time)

        # Check the constraint violation
        constraint_violation = check_constraint(maxwell)
        constraints.append(constraint_violation)

        # Update maxwell fields with the new values
        maxwell.E_x, maxwell.E_y, maxwell.E_z, maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi = fields

    return np.array(constraints)




def initialize(maxwell, A, lambda_):
    '''Initialize the electric field in the Maxwell object using a dipolar solution.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, containing the following attributes:
        - `x`, `y`, `z`: 3D arrays representing the mesh grid coordinates.
        - `r`: 3D array representing the radial distance from the origin.
        - `E_x`, `E_y`, `E_z`: 3D arrays that will store the initialized components of the electric field.
        - `A_x`, `A_y`, `A_z`: 3D arrays that will store the vector potential components (assumed to be zero initially).
    A : float
        The amplitude of the dipolar electric field solution.
    lambda_ : float
        The characteristic length scale of the dipolar electric field solution.
    Returns:
    --------
    object
        The Maxwell object with its `E_x`, `E_y`, and `E_z` attributes initialized to a dipolar solution.
    '''
    
    # Extract the coordinates and radial distance
    x = maxwell.x
    y = maxwell.y
    z = maxwell.z
    r = maxwell.r
    
    # Calculate the angular component sin(theta) = sqrt(y^2 + z^2) / r
    sin_theta = sqrt(y**2 + z**2) / r
    
    # Calculate the dipolar electric field component E_phi
    E_phi = -8 * A * r * sin_theta / lambda_**2 * exp(-(r / lambda_)**2)
    
    # Initialize the electric field components
    maxwell.E_x = E_phi * (-y / sqrt(y**2 + z**2))  # E_phi * -sin(phi)
    maxwell.E_y = E_phi * (x * z / (r * sqrt(y**2 + z**2)))  # E_phi * cos(phi) * cos(theta)
    maxwell.E_z = E_phi * (x * y / (r * sqrt(y**2 + z**2)))  # E_phi * cos(phi) * sin(theta)
    
    return maxwell


try:
    targets = process_hdf5_to_tuple('13.14', 3)
    target = targets[0]
    maxwell = Maxwell(50, 2)
    maxwell = initialize(maxwell)
    assert np.allclose((maxwell.E_x, maxwell.E_y, maxwell.E_z,  maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi), target)

    target = targets[1]
    maxwell = Maxwell(20, 2)
    maxwell = initialize(maxwell)
    assert np.allclose((maxwell.E_x, maxwell.E_y, maxwell.E_z,  maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi), target)

    target = targets[2]
    maxwell = Maxwell(100, 2)
    maxwell = initialize(maxwell)
    assert np.allclose((maxwell.E_x, maxwell.E_y, maxwell.E_z,  maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e