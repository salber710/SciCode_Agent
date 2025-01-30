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


# Background: In computational physics, especially in simulations involving wave propagation, it is often necessary to
# implement boundary conditions that allow waves to exit the simulation domain without reflecting back into it. This is
# known as an outgoing-wave boundary condition. For a scalar field f on a 3D grid, the outgoing-wave boundary condition
# can be applied at the outer boundaries of the grid. The idea is to approximate the behavior of waves as they leave the
# domain, which can be done by assuming that the wave propagates outward at a speed proportional to the distance from the
# origin. Mathematically, this can be expressed as a condition on the time derivative of the field at the boundary, such
# that the change in the field is proportional to the radial distance and the field value itself. This helps in simulating
# open boundary conditions where the field can freely propagate out of the simulation domain.

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
    nx, ny, nz = f.shape
    delta = maxwell.delta

    # Apply outgoing wave boundary condition on the outer boundary
    # x-direction
    f_dot[-1, :, :] = -f[-1, :, :] / (maxwell.r[-1, :, :] + delta)
    f_dot[0, :, :] = -f[0, :, :] / (maxwell.r[0, :, :] + delta)

    # y-direction
    f_dot[:, -1, :] = -f[:, -1, :] / (maxwell.r[:, -1, :] + delta)
    f_dot[:, 0, :] = -f[:, 0, :] / (maxwell.r[:, 0, :] + delta)

    # z-direction
    f_dot[:, :, -1] = -f[:, :, -1] / (maxwell.r[:, :, -1] + delta)
    f_dot[:, :, 0] = -f[:, :, 0] / (maxwell.r[:, :, 0] + delta)

    return f_dot


# Background: In the context of electromagnetism, the Lorentz gauge is a condition applied to the potentials in Maxwell's equations
# to simplify the equations and ensure gauge invariance. The equations for the time derivatives of the fields in the Lorentz gauge
# involve the Laplacian and divergence operators. Specifically, the time derivative of the electric field E_i is influenced by the
# Laplacian of the vector potential A_i, the divergence of A, and the current density j_i. The time derivative of the vector potential
# A_i is related to the electric field and the gradient of the scalar potential phi. The time derivative of phi is related to the
# divergence of A. Boundary conditions are crucial for ensuring the physical accuracy of the simulation, with mirror symmetry applied
# at the z=0 plane and cylindrical symmetry at the x=0 and y=0 planes. Outgoing wave boundary conditions are applied at the outer
# boundaries to simulate open space.


def laplace(f, delta):
    """Approximate the Laplacian of a 3D array using finite differences."""
    return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
            np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6 * f) / delta**2

def divergence(A_x, A_y, A_z, delta):
    """Approximate the divergence of a vector field (A_x, A_y, A_z)."""
    dAx_dx = (np.roll(A_x, -1, axis=0) - np.roll(A_x, 1, axis=0)) / (2 * delta)
    dAy_dy = (np.roll(A_y, -1, axis=1) - np.roll(A_y, 1, axis=1)) / (2 * delta)
    dAz_dz = (np.roll(A_z, -1, axis=2) - np.roll(A_z, 1, axis=2)) / (2 * delta)
    return dAx_dx + dAy_dy + dAz_dz

def gradient(phi, delta):
    """Compute the gradient of a scalar field phi."""
    grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * delta)
    grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * delta)
    grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * delta)
    return grad_x, grad_y, grad_z

def symmetry(field, x_sym, y_sym, z_sym):
    """Apply symmetry conditions to the field."""
    field[0, :, :] *= x_sym
    field[:, 0, :] *= y_sym
    field[:, :, 0] *= z_sym
    return field

def outgoing_wave(maxwell, field_dot, field):
    """Apply outgoing wave boundary conditions."""
    nx, ny, nz = field.shape
    field_dot[:, :, -1] = field[:, :, -2]  # Example of simple outgoing wave condition
    return field_dot

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
    nx, ny, nz = E_x.shape

    # Initialize derivatives
    E_x_dot = np.zeros_like(E_x)
    E_y_dot = np.zeros_like(E_y)
    E_z_dot = np.zeros_like(E_z)
    A_x_dot = np.zeros_like(A_x)
    A_y_dot = np.zeros_like(A_y)
    A_z_dot = np.zeros_like(A_z)
    phi_dot = np.zeros_like(phi)

    # Compute Laplacian of A components
    lap_A_x = laplace(A_x, delta)
    lap_A_y = laplace(A_y, delta)
    lap_A_z = laplace(A_z, delta)

    # Compute divergence of A
    div_A = divergence(A_x, A_y, A_z, delta)

    # Compute gradient of phi
    grad_phi_x, grad_phi_y, grad_phi_z = gradient(phi, delta)

    # Compute time derivatives using the Lorentz gauge equations
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # ∂t E_i = -∇²A_i + ∇_i(∇·A) - 4πj_i
                E_x_dot[i, j, k] = -lap_A_x[i, j, k] + div_A[i, j, k] - grad_phi_x[i, j, k]
                E_y_dot[i, j, k] = -lap_A_y[i, j, k] + div_A[i, j, k] - grad_phi_y[i, j, k]
                E_z_dot[i, j, k] = -lap_A_z[i, j, k] + div_A[i, j, k] - grad_phi_z[i, j, k]

                # ∂t A_i = -E_i - ∇_i φ
                A_x_dot[i, j, k] = -E_x[i, j, k] - grad_phi_x[i, j, k]
                A_y_dot[i, j, k] = -E_y[i, j, k] - grad_phi_y[i, j, k]
                A_z_dot[i, j, k] = -E_z[i, j, k] - grad_phi_z[i, j, k]

                # ∂t φ = -∇·A
                phi_dot[i, j, k] = -div_A[i, j, k]

    # Apply boundary conditions
    # Inner boundary conditions
    E_x_dot = symmetry(E_x_dot, 1, -1, -1)
    E_y_dot = symmetry(E_y_dot, -1, 1, -1)
    E_z_dot = symmetry(E_z_dot, -1, -1, 1)
    A_x_dot = symmetry(A_x_dot, 1, -1, -1)
    A_y_dot = symmetry(A_y_dot, -1, 1, -1)
    A_z_dot = symmetry(A_z_dot, -1, -1, 1)
    phi_dot = symmetry(phi_dot, 1, 1, 1)

    # Outgoing wave boundary conditions
    E_x_dot = outgoing_wave(maxwell, E_x_dot, E_x)
    E_y_dot = outgoing_wave(maxwell, E_y_dot, E_y)
    E_z_dot = outgoing_wave(maxwell, E_z_dot, E_z)
    A_x_dot = outgoing_wave(maxwell, A_x_dot, A_x)
    A_y_dot = outgoing_wave(maxwell, A_y_dot, A_y)
    A_z_dot = outgoing_wave(maxwell, A_z_dot, A_z)
    phi_dot = outgoing_wave(maxwell, phi_dot, phi)

    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)



# Background: In numerical simulations, updating the fields based on their time derivatives is a crucial step in
# advancing the simulation in time. This is typically done using a time-stepping method, where the new field values
# are computed by adding a scaled increment of the time derivatives to the current field values. The scaling factor
# and time step size determine the magnitude of the update. This approach is commonly used in explicit time integration
# schemes such as the Euler method or higher-order methods like Runge-Kutta. The scaling factor can be particularly
# useful when implementing higher-order integrators, allowing for more accurate and stable updates.

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

    # Iterate over each field and its corresponding time derivative
    for field, field_dot in zip(fields, fields_dot):
        # Update the field using the formula: new_field = field + factor * dt * field_dot
        new_field = field + factor * dt * field_dot
        # Append the updated field to the list
        new_fields.append(new_field)

    return new_fields

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('13.10', 3)
target = targets[0]

maxwell = Maxwell(50,2)
x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
fields = (x,y,z,x,y,z, z*0)
fields_dot = derivatives(maxwell, fields)
factor  = 0.5
dt      = 0.1
assert np.allclose(update_fields(maxwell, fields, fields_dot, factor, dt), target)
target = targets[1]

maxwell = Maxwell(50,2)
x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
fields = (x,y,z,-y,x,z*0, z*0+1)
fields_dot = derivatives(maxwell, fields)
factor  = 0.3
dt      = 0.1
assert np.allclose(update_fields(maxwell, fields, fields_dot, factor, dt), target)
target = targets[2]

maxwell = Maxwell(50,2)
x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
fields = (x,y,z,x*0,-z,y, z*0+1)
fields_dot = derivatives(maxwell, fields)
factor  = 0.2
dt      = 0.1
assert np.allclose(update_fields(maxwell, fields, fields_dot, factor, dt), target)
