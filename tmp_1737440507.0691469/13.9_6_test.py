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


# Background: The gradient of the divergence of a vector field represents the spatial rate of change of the 
# divergence at each point in the field. In a discrete setting, directly calculating the gradient of the divergence 
# using finite differences ensures accuracy and reduces computation errors. The second-order finite difference method 
# is used to achieve this. For a vector field A with components (A_x, A_y, A_z), we first compute the divergence 
# and then directly take the gradient of this divergence without storing intermediate results. This approach provides 
# a more accurate representation by calculating the necessary derivatives simultaneously.


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

    # Compute the gradient of the divergence for the interior points using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Divergence at point (i, j, k)
                div = (
                    (A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) +
                    (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) +
                    (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta)
                )

                # Gradient of divergence calculation
                grad_div_x[i, j, k] = (div[i+1, j, k] - div[i-1, j, k]) / (2 * delta)
                grad_div_y[i, j, k] = (div[i, j+1, k] - div[i, j-1, k]) / (2 * delta)
                grad_div_z[i, j, k] = (div[i, j, k+1] - div[i, j, k-1]) / (2 * delta)

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

# Background: In numerical simulations on grid-based domains, applying boundary conditions is crucial for
# accurately representing physical phenomena. Symmetry and antisymmetry are common boundary conditions.
# Symmetry (x_sym = 1) implies that the field value or its derivative is the same on either side of the boundary,
# while antisymmetry (x_sym = -1) implies that the field value or its derivative is inverted across the boundary.
# In a 3D simulation grid, these conditions are applied to the "inner" boundary grids (e.g., at x=0, y=0, z=0)
# to ensure that the simulation behaves consistently with the physical or mathematical model being considered.
# The boundary values are adjusted based on the symmetry factors provided for each axis.

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

    nx, ny, nz = f_dot.shape
    
    # Apply symmetry conditions on the x=0 plane
    f_dot[0, :, :] = x_sym * f_dot[1, :, :]
    
    # Apply symmetry conditions on the y=0 plane
    f_dot[:, 0, :] = y_sym * f_dot[:, 1, :]
    
    # Apply symmetry conditions on the z=0 plane
    f_dot[:, :, 0] = z_sym * f_dot[:, :, 1]
    
    return f_dot


# Background: In numerical simulations, applying appropriate boundary conditions is crucial for capturing the correct
# physical behavior at the edges of the simulation domain. The outgoing-wave boundary condition is a common approach
# used in wave simulations to simulate open boundaries, allowing waves to exit the simulation domain without reflection.
# This is particularly important in electromagnetics and acoustics simulations, where reflections at the boundary
# can interfere with the solution. The condition is typically implemented by assuming that the field at the boundary
# is propagating outward at the speed of light (or other relevant wave speed), resulting in a relationship between
# the field and its derivative at the boundary. In this context, for a field f and its time derivative f_dot, the
# outgoing boundary condition can be expressed as:
#   f_dot = -c * (f - f_previous) / delta
# where c is the speed of the wave (often taken as 1 for simplicity in reduced units), delta is the grid spacing,
# and f_previous is the field value from the previous timestep. This condition is applied at the outer boundaries
# of the simulation grid.

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

    # Speed of wave propagation (assuming c=1 for reduced units)
    c = 1.0

    # Apply outgoing wave boundary condition on the outer boundaries
    for j in range(ny):
        for k in range(nz):
            # X outer boundary
            f_dot[nx-1, j, k] = -c * (f[nx-1, j, k] - f[nx-2, j, k]) / delta

    for i in range(nx):
        for k in range(nz):
            # Y outer boundary
            f_dot[i, ny-1, k] = -c * (f[i, ny-1, k] - f[i, ny-2, k]) / delta

    for i in range(nx):
        for j in range(ny):
            # Z outer boundary
            f_dot[i, j, nz-1] = -c * (f[i, j, nz-1] - f[i, j, nz-2]) / delta

    return f_dot



# Background: In the Lorentz gauge, the time evolution of electromagnetic fields is described by Maxwell's equations.
# These equations relate the time derivatives of the electric field components (E_x, E_y, E_z) and the vector potential
# components (A_x, A_y, A_z) in terms of spatial derivatives of the fields and external currents (j_i). Specifically,
# the time derivative of the electric field components involves the Laplacian of the vector potential components and
# divergence terms. The time derivative of the vector potential components includes the electric field components and
# the gradient of the scalar potential phi. Additionally, the time derivative of the scalar potential phi is related
# to the divergence of the vector potential. Since the simulation is performed in one octant, boundary conditions must
# be applied: mirror symmetry on the z=0 plane and cylindrical symmetry on the x=0 and y=0 planes. Moreover, an outgoing
# wave boundary condition is applied on the outer boundary to simulate open boundaries.


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
    nx, ny, nz = A_x.shape
    delta = maxwell.delta
    
    # Initialize the derivatives
    E_x_dot = zeros((nx, ny, nz))
    E_y_dot = zeros((nx, ny, nz))
    E_z_dot = zeros((nx, ny, nz))
    A_x_dot = zeros((nx, ny, nz))
    A_y_dot = zeros((nx, ny, nz))
    A_z_dot = zeros((nx, ny, nz))
    phi_dot = zeros((nx, ny, nz))
    
    # Compute derivatives in the interior using central differences
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Laplacian of A_i
                lap_A_x = (A_x[i+1, j, k] + A_x[i-1, j, k] - 2 * A_x[i, j, k] +
                           A_x[i, j+1, k] + A_x[i, j-1, k] - 2 * A_x[i, j, k] +
                           A_x[i, j, k+1] + A_x[i, j, k-1] - 2 * A_x[i, j, k]) / (delta**2)
                lap_A_y = (A_y[i+1, j, k] + A_y[i-1, j, k] - 2 * A_y[i, j, k] +
                           A_y[i, j+1, k] + A_y[i, j-1, k] - 2 * A_y[i, j, k] +
                           A_y[i, j, k+1] + A_y[i, j, k-1] - 2 * A_y[i, j, k]) / (delta**2)
                lap_A_z = (A_z[i+1, j, k] + A_z[i-1, j, k] - 2 * A_z[i, j, k] +
                           A_z[i, j+1, k] + A_z[i, j-1, k] - 2 * A_z[i, j, k] +
                           A_z[i, j, k+1] + A_z[i, j, k-1] - 2 * A_z[i, j, k]) / (delta**2)
                
                # Divergence of A
                div_A = ((A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) +
                         (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) +
                         (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta))
                
                # Compute time derivatives
                E_x_dot[i, j, k] = -lap_A_x + (A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) * div_A
                E_y_dot[i, j, k] = -lap_A_y + (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) * div_A
                E_z_dot[i, j, k] = -lap_A_z + (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta) * div_A
                
                A_x_dot[i, j, k] = -E_x[i, j, k] - (phi[i+1, j, k] - phi[i-1, j, k]) / (2 * delta)
                A_y_dot[i, j, k] = -E_y[i, j, k] - (phi[i, j+1, k] - phi[i, j-1, k]) / (2 * delta)
                A_z_dot[i, j, k] = -E_z[i, j, k] - (phi[i, j, k+1] - phi[i, j, k-1]) / (2 * delta)
                
                phi_dot[i, j, k] = -div_A
    
    # Apply boundary conditions
    symmetry(E_x_dot, -1, -1, 1)
    symmetry(E_y_dot, -1, -1, 1)
    symmetry(E_z_dot, -1, -1, 1)
    symmetry(A_x_dot, 1, 1, 1)
    symmetry(A_y_dot, 1, 1, 1)
    symmetry(A_z_dot, 1, 1, 1)
    symmetry(phi_dot, 1, 1, 1)
    
    outgoing_wave(maxwell, E_x_dot, E_x)
    outgoing_wave(maxwell, E_y_dot, E_y)
    outgoing_wave(maxwell, E_z_dot, E_z)
    outgoing_wave(maxwell, A_x_dot, A_x)
    outgoing_wave(maxwell, A_y_dot, A_y)
    outgoing_wave(maxwell, A_z_dot, A_z)
    outgoing_wave(maxwell, phi_dot, phi)
    
    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)

from scicode.parse.parse import process_hdf5_to_tuple
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
