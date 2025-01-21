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


# Background: In the context of electromagnetism and numerical simulations, calculating the time derivatives 
# of the fields involved in Maxwell's equations is essential for simulating the dynamic behavior of electromagnetic 
# fields. The Lorentz gauge is a condition used to simplify Maxwell's equations, leading to the following forms:
# - The time derivative of the electric field components (E_i) involves the Laplacian (∇²) of the vector potential 
#   components (A_i), the divergence (∇·A), and any current density (j_i).
# - The time derivative of the vector potential components (A_i) is related to the electric field components and the 
#   gradient of the scalar potential (φ).
# - The time derivative of the scalar potential (φ) involves the divergence of the vector potential.
# Boundary conditions are applied to ensure correct simulation behavior: mirror symmetry on the z=0 plane 
# (reflecting the field values), and cylindrical symmetry on the x=0 and y=0 planes (imposing zero tangential 
# electric field and normal magnetic field conditions). Outgoing wave boundary conditions are applied to the 
# outer boundaries to allow waves to exit the simulation domain without reflection.


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

    # Initialize the derivative arrays
    E_x_dot = zeros((nx, ny, nz))
    E_y_dot = zeros((nx, ny, nz))
    E_z_dot = zeros((nx, ny, nz))
    A_x_dot = zeros((nx, ny, nz))
    A_y_dot = zeros((nx, ny, nz))
    A_z_dot = zeros((nx, ny, nz))
    phi_dot = zeros((nx, ny, nz))
    
    # Compute the Laplacian and Divergence for the derivatives
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # Laplacian of A_i
                laplace_A_x = (
                    (A_x[i+1, j, k] + A_x[i-1, j, k] - 2 * A_x[i, j, k]) +
                    (A_x[i, j+1, k] + A_x[i, j-1, k] - 2 * A_x[i, j, k]) +
                    (A_x[i, j, k+1] + A_x[i, j, k-1] - 2 * A_x[i, j, k])
                ) / (delta**2)
                
                laplace_A_y = (
                    (A_y[i+1, j, k] + A_y[i-1, j, k] - 2 * A_y[i, j, k]) +
                    (A_y[i, j+1, k] + A_y[i, j-1, k] - 2 * A_y[i, j, k]) +
                    (A_y[i, j, k+1] + A_y[i, j, k-1] - 2 * A_y[i, j, k])
                ) / (delta**2)
                
                laplace_A_z = (
                    (A_z[i+1, j, k] + A_z[i-1, j, k] - 2 * A_z[i, j, k]) +
                    (A_z[i, j+1, k] + A_z[i, j-1, k] - 2 * A_z[i, j, k]) +
                    (A_z[i, j, k+1] + A_z[i, j, k-1] - 2 * A_z[i, j, k])
                ) / (delta**2)

                # Divergence of A
                div_A = (
                    (A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) +
                    (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) +
                    (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta)
                )

                # Time derivative of E_i
                E_x_dot[i, j, k] = -laplace_A_x + (A_x[i, j+1, k] - A_x[i, j-1, k]) / (2 * delta) + (A_x[i, j, k+1] - A_x[i, j, k-1]) / (2 * delta) - 4 * np.pi * 0  # Assuming j_i = 0 for simplicity
                E_y_dot[i, j, k] = -laplace_A_y + (A_y[i+1, j, k] - A_y[i-1, j, k]) / (2 * delta) + (A_y[i, j, k+1] - A_y[i, j, k-1]) / (2 * delta) - 4 * np.pi * 0  # Assuming j_i = 0 for simplicity
                E_z_dot[i, j, k] = -laplace_A_z + (A_z[i+1, j, k] - A_z[i-1, j, k]) / (2 * delta) + (A_z[i, j+1, k] - A_z[i, j-1, k]) / (2 * delta) - 4 * np.pi * 0  # Assuming j_i = 0 for simplicity

                # Time derivative of A_i
                A_x_dot[i, j, k] = -E_x[i, j, k] - (phi[i+1, j, k] - phi[i-1, j, k]) / (2 * delta)
                A_y_dot[i, j, k] = -E_y[i, j, k] - (phi[i, j+1, k] - phi[i, j-1, k]) / (2 * delta)
                A_z_dot[i, j, k] = -E_z[i, j, k] - (phi[i, j, k+1] - phi[i, j, k-1]) / (2 * delta)

                # Time derivative of phi
                phi_dot[i, j, k] = -div_A

    # Apply boundary conditions
    for axis_sym, sym in zip((E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot), 
                             ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (1, 1, 1))):
        symmetry(axis_sym, *sym)
        outgoing_wave(maxwell, axis_sym, fields[sym.index(1)])

    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


# Background: In numerical simulations, updating fields based on their time derivatives is a crucial step in 
# time integration schemes. This process involves incrementally advancing the field values in time by adding 
# a scaled version of the time derivatives to the current fields. The scale is determined by the product of a 
# time step size (dt) and an update factor (factor), which can be useful when implementing higher-order time 
# integration methods such as Runge-Kutta. In this function, each field is updated independently by adding 
# the product of its corresponding time derivative, the scaling factor, and the time step size to its current value.

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

    # Create a list to store the updated fields
    new_fields = []

    # Update each field by adding the scaled time derivative
    for field, field_dot in zip(fields, fields_dot):
        # Compute the update as the product of the time derivative, factor, and dt
        update = factor * dt * field_dot
        # Update the field by adding the computed update
        new_field = field + update
        # Append the updated field to the list
        new_fields.append(new_field)

    return new_fields


# Background: The Crank-Nicholson method is a numerical technique used for solving partial differential equations 
# that is unconditionally stable and second-order accurate in both time and space. It is often used for time-stepping
# in simulations of wave equations, such as those governed by Maxwell's equations. The method is implicit, requiring 
# iterative solutions at each time step. In this context, we're using an iterative Crank-Nicholson scheme, which involves 
# multiple substeps to converge to a stable solution at each time step. The Courant number provides a criterion for stability, 
# ensuring that the time step size is small enough relative to the spatial grid size to maintain numerical stability. The 
# larger time integration is achieved by breaking it into smaller substeps controlled by the Courant condition.

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

    # Calculate the substep size using the Courant number
    delta_t = courant * maxwell.delta / np.sqrt(3)  # sqrt(3) for 3D stability
    n_steps = int(np.ceil(t_const / delta_t))
    delta_t = t_const / n_steps  # Adjust delta_t to divide t_const evenly

    # Copy the initial fields to avoid in-place updates
    new_fields = [np.copy(field) for field in fields]

    for _ in range(n_steps):
        # Compute derivatives using the current fields
        fields_dot = derivatives(maxwell, new_fields)

        # First half-step
        half_step_fields = update_fields(maxwell, new_fields, fields_dot, 0.5, delta_t)
        
        # Compute derivatives at the half-step
        fields_dot_half = derivatives(maxwell, half_step_fields)

        # Update fields using the derivatives at the half-step
        new_fields = update_fields(maxwell, new_fields, fields_dot_half, 1.0, delta_t)

    # Update the simulation time
    maxwell.t += t_const

    return maxwell.t, new_fields


# Background: In the context of 3+1 formalism for Maxwell's equations, constraint violations are important 
# indicators of numerical accuracy and stability. Specifically, the constraint equation D_i E^i - 4πρ = 0 
# should hold true at each time slice in a source-free evolution, where ρ = 0. This means that the divergence 
# of the electric field should ideally be zero throughout the grid. However, numerical errors can accumulate 
# over time, leading to constraint violations. To quantify these violations, we compute the divergence of the 
# electric field components and evaluate the L2 norm of these values. The L2 norm provides a measure of the 
# magnitude of constraint violations across the entire grid.

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

    E_x = maxwell.E_x
    E_y = maxwell.E_y
    E_z = maxwell.E_z
    delta = maxwell.delta

    # Get the shape of the grid
    nx, ny, nz = E_x.shape

    # Initialize the divergence array
    div_E = zeros((nx, ny, nz))

    # Calculate the divergence of the electric field at the interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                div_E[i, j, k] = (
                    (E_x[i+1, j, k] - E_x[i-1, j, k]) / (2 * delta) +
                    (E_y[i, j+1, k] - E_y[i, j-1, k]) / (2 * delta) +
                    (E_z[i, j, k+1] - E_z[i, j, k-1]) / (2 * delta)
                )

    # Compute the L2 norm of the constraint violation
    norm_c = sqrt(np.sum(div_E**2) * delta**3)

    return norm_c


# Background: In numerical simulations of time-dependent systems, such as those governed by Maxwell's equations,
# it is crucial to accurately integrate the fields over time. This involves repeatedly applying a time-stepping
# method, like the iterative Crank-Nicholson scheme, to evolve the fields from their initial state to a final
# time t_max. During this process, it is important to monitor the simulation for physical consistency by checking
# the constraint violations at regular intervals (every t_check time units). These checks help ensure that the
# numerical solution remains stable and accurate over the course of the integration. The function will perform
# these tasks, storing the constraint violations for analysis.

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

    constraints = []  # List to store constraint violations at each check interval
    current_time = maxwell.t  # Start from the current simulation time
    fields = [maxwell.E_x, maxwell.E_y, maxwell.E_z, maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi]

    while current_time < t_max:
        # Calculate the time step size based on the next check time or the remaining time to t_max
        next_check_time = min(current_time + t_check, t_max)
        step_time = next_check_time - current_time
        
        # Perform the integration step
        current_time, fields = stepper(maxwell, fields, courant, step_time)

        # Update the fields in the maxwell object
        maxwell.E_x, maxwell.E_y, maxwell.E_z, maxwell.A_x, maxwell.A_y, maxwell.A_z, maxwell.phi = fields

        # Check and record the constraint violation
        constraint_violation = check_constraint(maxwell)
        constraints.append(constraint_violation)

    return np.array(constraints)



# Background: In computational physics, initializing fields according to known analytical solutions is a common
# practice to ensure that simulations start with physically meaningful configurations. In the context of
# electromagnetic simulations, a dipolar electric field is a well-known solution that captures the essence of
# electric fields generated by dipoles. The given dipolar electric field expression, E_phi, is a function of the
# radial distance (r), angle (theta), and characteristic length scale (lambda). The expression involves exponential
# decay, which ensures that the field diminishes at large distances from the dipole source. The challenge is to
# represent this analytical solution in terms of Cartesian components (E_x, E_y, E_z) on a 3D grid, which requires
# transforming the spherical coordinates (r, theta, phi) into their Cartesian counterparts.

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
        The Maxwell object with its `E_x`, `E_y`, and `E_z` attributes initialized to a dipolar solution.
    '''

    # Constants for the dipolar electric field
    A = 1.0  # Amplitude of the dipole
    lambda_ = 1.0  # Characteristic length scale

    # Extract grid coordinates and radial distance
    x, y, z, r = maxwell.x, maxwell.y, maxwell.z, maxwell.r

    # Calculate theta (polar angle) from z and r
    theta = np.arccos(z / (r + np.finfo(float).eps))  # Add eps to avoid division by zero

    # Calculate the dipolar electric field components in spherical coordinates
    E_phi = -8 * A * (r * np.sin(theta)) / (lambda_**2) * np.exp(-(r / lambda_)**2)

    # Convert E_phi from spherical to Cartesian components
    # Using spherical to Cartesian transformation for E_phi
    E_x = -E_phi * y / (np.sqrt(x**2 + y**2) + np.finfo(float).eps)
    E_y = E_phi * x / (np.sqrt(x**2 + y**2) + np.finfo(float).eps)
    E_z = zeros(x.shape)  # E_phi does not contribute to E_z in this dipolar configuration

    # Initialize the Maxwell object fields
    maxwell.E_x = E_x
    maxwell.E_y = E_y
    maxwell.E_z = E_z

    return maxwell

from scicode.parse.parse import process_hdf5_to_tuple
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
