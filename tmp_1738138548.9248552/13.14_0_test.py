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



def laplace(fct, delta):
    nx, ny, nz = fct.shape
    lap = np.zeros_like(fct)
    delta_sq = delta ** 2

    # Compute the Laplacian using a combination of slicing and numpy pad for boundary handling
    padded_fct = np.pad(fct, pad_width=1, mode='constant', constant_values=0)
    lap[1:-1, 1:-1, 1:-1] = (
        padded_fct[2:, 1:-1, 1:-1] + padded_fct[:-2, 1:-1, 1:-1] +
        padded_fct[1:-1, 2:, 1:-1] + padded_fct[1:-1, :-2, 1:-1] +
        padded_fct[1:-1, 1:-1, 2:] + padded_fct[1:-1, 1:-1, :-2] -
        6 * padded_fct[1:-1, 1:-1, 1:-1]
    ) / delta_sq

    return lap



def gradient(fct, delta):
    nx, ny, nz = fct.shape
    grad_x = np.zeros_like(fct)
    grad_y = np.zeros_like(fct)
    grad_z = np.zeros_like(fct)

    # Compute gradients using second-order central differences
    # Use np.roll for circular shift and handle boundaries separately
    fct_x_plus = np.roll(fct, -1, axis=0)
    fct_x_minus = np.roll(fct, 1, axis=0)
    fct_y_plus = np.roll(fct, -1, axis=1)
    fct_y_minus = np.roll(fct, 1, axis=1)
    fct_z_plus = np.roll(fct, -1, axis=2)
    fct_z_minus = np.roll(fct, 1, axis=2)

    grad_x[1:-1, :, :] = (fct_x_plus[1:-1, :, :] - fct_x_minus[1:-1, :, :]) / (2 * delta)
    grad_y[:, 1:-1, :] = (fct_y_plus[:, 1:-1, :] - fct_y_minus[:, 1:-1, :]) / (2 * delta)
    grad_z[:, :, 1:-1] = (fct_z_plus[:, :, 1:-1] - fct_z_minus[:, :, 1:-1]) / (2 * delta)

    # Zero out the boundary values explicitly
    grad_x[0, :, :] = grad_x[-1, :, :] = 0
    grad_y[:, 0, :] = grad_y[:, -1, :] = 0
    grad_z[:, :, 0] = grad_z[:, :, -1] = 0

    return grad_x, grad_y, grad_z


def divergence(v_x, v_y, v_z, delta):


    # Initialize the divergence array with zeros
    div = np.zeros_like(v_x)

    # Compute divergence using second-order central differences
    # Apply vectorized operations across shifted slices
    div[1:-1, 1:-1, 1:-1] = (
        (v_x[2:, 1:-1, 1:-1] - v_x[0:-2, 1:-1, 1:-1]) +
        (v_y[1:-1, 2:, 1:-1] - v_y[1:-1, 0:-2, 1:-1]) +
        (v_z[1:-1, 1:-1, 2:] - v_z[1:-1, 1:-1, 0:-2])
    ) / (2 * delta)

    # Ensure the boundary values remain zero
    return div



def grad_div(A_x, A_y, A_z, delta):
    nx, ny, nz = A_x.shape
    grad_div_x = np.zeros_like(A_x)
    grad_div_y = np.zeros_like(A_y)
    grad_div_z = np.zeros_like(A_z)

    # Compute the gradient of the divergence using second-order central differences
    # Compute each component's contribution using vectorized operations for efficiency
    # Adjusting the computation to ensure each component is calculated distinctly
    # Using a more compact vectorized form for better performance and readability
    grad_div_x[1:-1, 1:-1, 1:-1] = (
        (A_x[2:, 1:-1, 1:-1] - 2 * A_x[1:-1, 1:-1, 1:-1] + A_x[:-2, 1:-1, 1:-1]) +
        (A_y[1:-1, 2:, 1:-1] - A_y[1:-1, :-2, 1:-1]) +
        (A_z[1:-1, 1:-1, 2:] - A_z[1:-1, 1:-1, :-2])
    ) / (delta ** 2)

    grad_div_y[1:-1, 1:-1, 1:-1] = (
        (A_x[1:-1, 2:, 1:-1] - A_x[1:-1, :-2, 1:-1]) +
        (A_y[2:, 1:-1, 1:-1] - 2 * A_y[1:-1, 1:-1, 1:-1] + A_y[:-2, 1:-1, 1:-1]) +
        (A_z[1:-1, 1:-1, 2:] - A_z[1:-1, 1:-1, :-2])
    ) / (delta ** 2)

    grad_div_z[1:-1, 1:-1, 1:-1] = (
        (A_x[1:-1, 1:-1, 2:] - A_x[1:-1, 1:-1, :-2]) +
        (A_y[1:-1, 1:-1, 2:] - A_y[1:-1, 1:-1, :-2]) +
        (A_z[2:, 1:-1, 1:-1] - 2 * A_z[1:-1, 1:-1, 1:-1] + A_z[:-2, 1:-1, 1:-1])
    ) / (delta ** 2)

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
    # Apply symmetry conditions using a combination of advanced slicing and in-place multiplication
    # This method focuses on using advanced numpy techniques for efficient in-place operations

    # Apply symmetry for x=0 boundary
    f_dot[0, :, :] = x_sym * f_dot[1, :, :].copy()

    # Apply symmetry for y=0 boundary
    f_dot[:, 0, :] = y_sym * f_dot[:, 1, :].copy()

    # Apply symmetry for z=0 boundary
    f_dot[:, :, 0] = z_sym * f_dot[:, :, 1].copy()

    return f_dot


def outgoing_wave(maxwell, f_dot, f):


    # Constants
    c = 1.0  # Speed of wave propagation

    # Extract grid properties
    delta = maxwell.delta
    r = maxwell.r

    # Compute the radial gradient using spherical coordinates approximation
    radial_gradient = np.zeros_like(f)
    radial_gradient[:-1, :, :] = (f[1:, :, :] - f[:-1, :, :]) / delta
    radial_gradient[:, :-1, :] += (f[:, 1:, :] - f[:, :-1, :]) / delta
    radial_gradient[:, :, :-1] += (f[:, :, 1:] - f[:, :, :-1]) / delta

    # Calculate the radial component of the gradient
    radial_component = (radial_gradient * maxwell.x / r + radial_gradient * maxwell.y / r + radial_gradient * maxwell.z / r)

    # Compute a damping factor that increases with radial distance
    damping = np.exp(-0.1 * (r - np.min(r)))

    # Apply the outgoing wave boundary condition with damping
    # Update f_dot at the boundaries based on the radial component, speed of wave propagation, and damping
    boundary_mask = (r >= np.max(r) * 0.95)  # Mask for the outer 5% of the radial distances
    f_dot[boundary_mask] = -c * radial_component[boundary_mask] * damping[boundary_mask]

    return f_dot


def derivatives(maxwell, fields):


    # Unpack fields
    E_x, E_y, E_z, A_x, A_y, A_z, phi = fields
    delta = maxwell.delta
    j_x, j_y, j_z = maxwell.current_density

    # Helper functions for finite differences using central differences
    def central_diff(f, axis):
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * delta)

    def laplacian(f):
        return (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
                np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) +
                np.roll(f, -1, axis=2) + np.roll(f, 1, axis=2) - 6 * f) / delta**2

    # Compute derivatives
    div_A = central_diff(A_x, 0) + central_diff(A_y, 1) + central_diff(A_z, 2)
    lap_A_x = laplacian(A_x)
    lap_A_y = laplacian(A_y)
    lap_A_z = laplacian(A_z)

    E_x_dot = -lap_A_x + central_diff(div_A, 0) - 4 * np.pi * j_x
    E_y_dot = -lap_A_y + central_diff(div_A, 1) - 4 * np.pi * j_y
    E_z_dot = -lap_A_z + central_diff(div_A, 2) - 4 * np.pi * j_z

    A_x_dot = -E_x - central_diff(phi, 0)
    A_y_dot = -E_y - central_diff(phi, 1)
    A_z_dot = -E_z - central_diff(phi, 2)

    phi_dot = -div_A

    # Apply boundary conditions
    # Mirror symmetry at z=0
    E_x_dot[:, :, 0] = E_x_dot[:, :, 1]
    E_y_dot[:, :, 0] = E_y_dot[:, :, 1]
    E_z_dot[:, :, 0] = -E_z_dot[:, :, 1]
    A_x_dot[:, :, 0] = A_x_dot[:, :, 1]
    A_y_dot[:, :, 0] = A_y_dot[:, :, 1]
    A_z_dot[:, :, 0] = -A_z_dot[:, :, 1]
    phi_dot[:, :, 0] = phi_dot[:, :, 1]

    # Cylindrical symmetry at x=0 and y=0
    E_x_dot[0, :, :] = -E_x_dot[1, :, :]
    E_y_dot[:, 0, :] = -E_y_dot[:, 1, :]
    A_x_dot[0, :, :] = -A_x_dot[1, :, :]
    A_y_dot[:, 0, :] = -A_y_dot[:, 1, :]
    phi_dot[0, :, :] = phi_dot[1, :, :]
    phi_dot[:, 0, :] = phi_dot[:, 1, :]

    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


def update_fields(maxwell, fields, fields_dot, factor, dt):
    # Use a vectorized approach with numpy's einsum for optimal performance and clarity


    # Calculate the update for each field using einsum for element-wise multiplication and addition
    new_fields = [np.einsum('ijk,ijk->ijk', field, 1 + factor * dt * field_dot) for field, field_dot in zip(fields, fields_dot)]

    return new_fields



def crank_nicholson_step(maxwell, fields, dt):
    """
    Perform a single Crank-Nicholson step using a quadratic extrapolation method.
    Parameters:
    -----------
    maxwell : object
        Contains simulation parameters and methods for calculating field derivatives.
    fields : list of numpy.ndarray
        Current field values.
    dt : float
        Time step size for the update.
    
    Returns:
    --------
    list of numpy.ndarray
        Updated field values after one Crank-Nicholson step.
    """
    n_fields = len(fields)
    new_fields = [np.zeros_like(field) for field in fields]
    field_derivatives = maxwell.calculate_field_derivatives(fields)
    
    # Predictor step using current derivatives
    predictor_fields = [fields[i] + dt * field_derivatives[i] for i in range(n_fields)]
    predictor_derivatives = maxwell.calculate_field_derivatives(predictor_fields)
    
    # Corrector step using a quadratic extrapolation of the derivatives
    for i in range(n_fields):
        new_fields[i] = fields[i] + dt * (field_derivatives[i] + 4 * predictor_derivatives[i]) / 5
    
    return new_fields

def iterative_crank_nicholson(maxwell, fields, courant, t_const):
    """
    Perform multiple Crank-Nicholson steps to advance the simulation by t_const using a quadratic extrapolation method.
    Parameters:
    -----------
    maxwell : object
        Contains simulation parameters and methods for calculating field derivatives.
    fields : list of numpy.ndarray
        Current field values.
    courant : float
        Courant number for stability control.
    t_const : float
        Total time to advance the simulation.
    
    Returns:
    --------
    tuple (float, list of numpy.ndarray)
        Updated simulation time and field values.
    """
    delta_x = maxwell.delta
    c = maxwell.speed_of_light
    delta_t = courant * delta_x / c
    
    num_steps = int(np.ceil(t_const / delta_t))
    actual_delta_t = t_const / num_steps
    
    current_fields = fields
    for _ in range(num_steps):
        current_fields = crank_nicholson_step(maxwell, current_fields, actual_delta_t)
    
    maxwell.t += t_const
    return maxwell.t, current_fields



def check_constraint(maxwell):
    E_x, E_y, E_z = maxwell.E_x, maxwell.E_y, maxwell.E_z
    delta = maxwell.delta

    # Compute divergence using a combination of central and one-sided differences with boundary handling
    div_E = np.zeros_like(E_x)
    # Central differences for the bulk of the grid
    div_E[1:-1, 1:-1, 1:-1] = (
        (E_x[2:, 1:-1, 1:-1] - E_x[:-2, 1:-1, 1:-1]) +
        (E_y[1:-1, 2:, 1:-1] - E_y[1:-1, :-2, 1:-1]) +
        (E_z[1:-1, 1:-1, 2:] - E_z[1:-1, 1:-1, :-2])
    ) / (2 * delta)

    # One-sided differences for the boundaries
    # X boundaries
    div_E[0, :, :] = (E_x[1, :, :] - E_x[0, :, :]) / delta
    div_E[-1, :, :] = (E_x[-1, :, :] - E_x[-2, :, :]) / delta
    # Y boundaries
    div_E[:, 0, :] = (E_y[:, 1, :] - E_y[:, 0, :]) / delta
    div_E[:, -1, :] = (E_y[:, -1, :] - E_y[:, -2, :]) / delta
    # Z boundaries
    div_E[:, :, 0] = (E_z[:, :, 1] - E_z[:, :, 0]) / delta
    div_E[:, :, -1] = (E_z[:, :, -1] - E_z[:, :, -2]) / delta

    # Calculate the L2 norm of the constraint violation
    norm_c = np.sqrt(np.sum(div_E**2) * delta**3)

    return norm_c


def integrate(maxwell, courant, t_max, t_check):


    constraints = []
    dt = courant * maxwell.delta * np.tan(np.pi / 8)  # Using tangent of pi/8 for time step adjustment
    current_time = maxwell.t
    next_check_time = current_time + t_check

    while current_time < t_max:
        step(maxwell, dt)
        current_time += dt
        maxwell.t = current_time

        if current_time >= next_check_time:
            constraints.append(check_constraint(maxwell))
            next_check_time += t_check

    return np.array(constraints)



# Background: In electromagnetism, a dipolar electric field is a field configuration that resembles the field
# produced by a dipole, which consists of two equal and opposite charges separated by a distance. The dipolar
# field in spherical coordinates can be expressed in terms of the radial distance `r`, the polar angle `theta`,
# and the azimuthal angle `phi`. For a dipolar electric field, the component `E_phi` is often of interest,
# especially in problems involving spherical symmetry. The given expression for `E_phi` involves an exponential
# decay factor that depends on the radial distance `r` and a characteristic length scale `lambda`. The task is
# to initialize the electric field components `E_x`, `E_y`, and `E_z` in a Cartesian coordinate system using
# the provided dipolar solution for `E_phi`.

def initialize(maxwell):
    '''Initialize the electric field in the Maxwell object using a dipolar solution.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment, containing the following attributes:
        - `x`, `y`, `z`: 3D arrays representing the mesh grid coordinates.
        - `r`: 3D array representing the radial distance from the origin.
        - `E_x`, `E_y`, `E_z`: 3D arrays that will store the initialized components of the electric field.
        - `A_x`, `A_y`, `A_z`: 3D arrays that will store the vector potential components (assumed to be zero initially).
    Returns:
    --------
    object
        The Maxwell object with its `E_x` and `E_y` attributes initialized to a dipolar solution.
    '''
    # Constants
    A = 1.0  # Amplitude of the dipolar field
    lambda_ = 1.0  # Characteristic length scale

    # Calculate theta in spherical coordinates
    theta = np.arccos(maxwell.z / maxwell.r)

    # Calculate the dipolar electric field component E_phi
    E_phi = -8 * A * (maxwell.r * np.sin(theta) / lambda_**2) * np.exp(-(maxwell.r / lambda_)**2)

    # Convert E_phi to Cartesian components E_x, E_y, E_z
    # E_phi is perpendicular to the radial vector in the xy-plane
    # E_x = -E_phi * y / sqrt(x^2 + y^2)
    # E_y = E_phi * x / sqrt(x^2 + y^2)
    # E_z = 0 (since E_phi is in the xy-plane)

    # Avoid division by zero by setting a small threshold
    xy_norm = np.sqrt(maxwell.x**2 + maxwell.y**2)
    xy_norm[xy_norm == 0] = 1e-10  # Avoid division by zero

    maxwell.E_x = -E_phi * maxwell.y / xy_norm
    maxwell.E_y = E_phi * maxwell.x / xy_norm
    maxwell.E_z = np.zeros_like(maxwell.E_x)  # E_phi has no z-component

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