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



# Background: In the context of electromagnetic field simulations, the Lorentz gauge is a condition that simplifies Maxwell's equations. 
# It is given by the equation ∂_i A^i + ∂_t φ = 0, where A^i is the vector potential and φ is the scalar potential. 
# The time derivatives of the electric field E_i and the vector potential A_i are derived from Maxwell's equations under this gauge. 
# The equations for the time derivatives are:
# ∂_t E_i = -∇^2 A_i + ∇_i (∇_j A^j) - 4π j_i
# ∂_t A_i = -E_i - ∇_i φ
# ∂_t φ = -∇_i A^i
# These equations describe how the fields evolve over time. The boundary conditions are crucial for ensuring the physical accuracy of the simulation. 
# Mirror symmetry is applied across the z=0 plane, and cylindrical symmetry is applied across the x=0 and y=0 planes. 
# An outgoing wave boundary condition is applied at the outer boundary to simulate open space.

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


    # Unpack fields
    E_x, E_y, E_z, A_x, A_y, A_z, phi = fields
    delta = maxwell.delta

    # Initialize derivatives
    E_x_dot = zeros_like(E_x)
    E_y_dot = zeros_like(E_y)
    E_z_dot = zeros_like(E_z)
    A_x_dot = zeros_like(A_x)
    A_y_dot = zeros_like(A_y)
    A_z_dot = zeros_like(A_z)
    phi_dot = zeros_like(phi)

    # Compute Laplacian of A_i
    lap_A_x = laplace(A_x, delta)
    lap_A_y = laplace(A_y, delta)
    lap_A_z = laplace(A_z, delta)

    # Compute divergence of A_i
    div_A = divergence(A_x, A_y, A_z, delta)

    # Compute gradient of phi
    grad_phi_x, grad_phi_y, grad_phi_z = gradient(phi, delta)

    # Compute time derivatives
    E_x_dot[1:-1, 1:-1, 1:-1] = -lap_A_x[1:-1, 1:-1, 1:-1] + gradient(div_A, delta)[0][1:-1, 1:-1, 1:-1]
    E_y_dot[1:-1, 1:-1, 1:-1] = -lap_A_y[1:-1, 1:-1, 1:-1] + gradient(div_A, delta)[1][1:-1, 1:-1, 1:-1]
    E_z_dot[1:-1, 1:-1, 1:-1] = -lap_A_z[1:-1, 1:-1, 1:-1] + gradient(div_A, delta)[2][1:-1, 1:-1, 1:-1]

    A_x_dot[1:-1, 1:-1, 1:-1] = -E_x[1:-1, 1:-1, 1:-1] - grad_phi_x[1:-1, 1:-1, 1:-1]
    A_y_dot[1:-1, 1:-1, 1:-1] = -E_y[1:-1, 1:-1, 1:-1] - grad_phi_y[1:-1, 1:-1, 1:-1]
    A_z_dot[1:-1, 1:-1, 1:-1] = -E_z[1:-1, 1:-1, 1:-1] - grad_phi_z[1:-1, 1:-1, 1:-1]

    phi_dot[1:-1, 1:-1, 1:-1] = -div_A[1:-1, 1:-1, 1:-1]

    # Apply boundary conditions
    E_x_dot = symmetry(E_x_dot, -1, 1, 1)
    E_y_dot = symmetry(E_y_dot, 1, -1, 1)
    E_z_dot = symmetry(E_z_dot, 1, 1, -1)

    A_x_dot = symmetry(A_x_dot, -1, 1, 1)
    A_y_dot = symmetry(A_y_dot, 1, -1, 1)
    A_z_dot = symmetry(A_z_dot, 1, 1, -1)

    phi_dot = symmetry(phi_dot, 1, 1, 1)

    # Apply outgoing wave boundary condition
    E_x_dot = outgoing_wave(maxwell, E_x_dot, E_x)
    E_y_dot = outgoing_wave(maxwell, E_y_dot, E_y)
    E_z_dot = outgoing_wave(maxwell, E_z_dot, E_z)

    A_x_dot = outgoing_wave(maxwell, A_x_dot, A_x)
    A_y_dot = outgoing_wave(maxwell, A_y_dot, A_y)
    A_z_dot = outgoing_wave(maxwell, A_z_dot, A_z)

    phi_dot = outgoing_wave(maxwell, phi_dot, phi)

    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


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