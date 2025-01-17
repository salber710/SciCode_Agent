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

# Background: In computational physics, applying boundary conditions is crucial for ensuring that the numerical
# solution of a differential equation respects the physical constraints of the problem. In the context of a
# simulation on a 3D grid, symmetry boundary conditions are often used to reduce computational effort by
# exploiting the symmetry of the problem. For a scalar or vector field, symmetry (or antisymmetry) can be
# applied at the boundaries. Symmetric boundary conditions imply that the field is mirrored across the boundary,
# while antisymmetric conditions imply that the field is inverted. For example, if a field is symmetric across
# the x=0 plane, the value at x=0 is the same as at x=1, but if it is antisymmetric, the value at x=0 is the
# negative of the value at x=1. This function will apply such symmetry conditions to the inner boundary mesh
# grids of a 3D array representing time derivatives of a field.

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


# Background: In computational physics, applying boundary conditions is crucial for ensuring that the numerical
# solution of a differential equation respects the physical constraints of the problem. The outgoing wave boundary
# condition is used to simulate open boundaries where waves can exit the simulation domain without reflecting back.
# This is particularly important in electromagnetic simulations, such as those involving Maxwell's equations, to
# prevent artificial reflections from the boundaries. The outgoing wave boundary condition can be implemented by
# assuming that the field behaves like a wave propagating outward, which can be approximated by a first-order
# derivative in time and space. For a field f, the outgoing wave condition at the boundary can be expressed as:
# f_dot = -c * (f / r), where c is the speed of the wave (often set to 1 in normalized units), and r is the radial
# distance from the origin. This condition is applied at the outer boundaries of the simulation grid.

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
    c = 1.0  # Speed of the wave, often set to 1 in normalized units

    # Apply outgoing wave boundary condition on the outer boundaries
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i == nx - 1 or j == ny - 1 or k == nz - 1:
                    r = maxwell.r[i, j, k]
                    if r != 0:
                        f_dot[i, j, k] = -c * (f[i, j, k] / r)

    return f_dot


# Background: In the context of electromagnetic simulations, the time evolution of the fields is governed by Maxwell's equations.
# In the Lorentz gauge, these equations can be expressed in terms of the vector potential A and the scalar potential φ.
# The time derivative of the electric field E is influenced by the Laplacian of the vector potential A, the divergence of A,
# and the current density j. The time derivative of the vector potential A is related to the electric field E and the gradient
# of the scalar potential φ. The time derivative of the scalar potential φ is determined by the divergence of A.
# Boundary conditions are crucial for ensuring the physical accuracy of the simulation. Mirror symmetry is applied across the z=0 plane,
# while cylindrical symmetry is applied across the x=0 and y=0 planes. Outgoing wave boundary conditions are applied at the outer boundaries
# to simulate open boundaries where waves can exit the simulation domain without reflecting back.

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

    # Initialize derivatives
    E_x_dot = np.zeros_like(E_x)
    E_y_dot = np.zeros_like(E_y)
    E_z_dot = np.zeros_like(E_z)
    A_x_dot = np.zeros_like(A_x)
    A_y_dot = np.zeros_like(A_y)
    A_z_dot = np.zeros_like(A_z)
    phi_dot = np.zeros_like(phi)

    # Compute Laplacian and divergence
    lap_A_x = laplace(A_x, delta)
    lap_A_y = laplace(A_y, delta)
    lap_A_z = laplace(A_z, delta)
    div_A = divergence(A_x, A_y, A_z, delta)

    # Compute time derivatives
    for i in range(1, E_x.shape[0] - 1):
        for j in range(1, E_x.shape[1] - 1):
            for k in range(1, E_x.shape[2] - 1):
                # ∂t E_i = -∇²A_i + ∇i(∇·A) - 4πj_i
                E_x_dot[i, j, k] = -lap_A_x[i, j, k] + (A_x[i+1, j, k] - A_x[i-1, j, k]) / (2 * delta) * div_A[i, j, k]
                E_y_dot[i, j, k] = -lap_A_y[i, j, k] + (A_y[i, j+1, k] - A_y[i, j-1, k]) / (2 * delta) * div_A[i, j, k]
                E_z_dot[i, j, k] = -lap_A_z[i, j, k] + (A_z[i, j, k+1] - A_z[i, j, k-1]) / (2 * delta) * div_A[i, j, k]

                # ∂t A_i = -E_i - ∇iφ
                A_x_dot[i, j, k] = -E_x[i, j, k] - (phi[i+1, j, k] - phi[i-1, j, k]) / (2 * delta)
                A_y_dot[i, j, k] = -E_y[i, j, k] - (phi[i, j+1, k] - phi[i, j-1, k]) / (2 * delta)
                A_z_dot[i, j, k] = -E_z[i, j, k] - (phi[i, j, k+1] - phi[i, j, k-1]) / (2 * delta)

                # ∂t φ = -∇·A
                phi_dot[i, j, k] = -div_A[i, j, k]

    # Apply boundary conditions
    E_x_dot = symmetry(E_x_dot, -1, 1, 1)
    E_y_dot = symmetry(E_y_dot, 1, -1, 1)
    E_z_dot = symmetry(E_z_dot, 1, 1, -1)

    A_x_dot = symmetry(A_x_dot, -1, 1, 1)
    A_y_dot = symmetry(A_y_dot, 1, -1, 1)
    A_z_dot = symmetry(A_z_dot, 1, 1, -1)

    phi_dot = symmetry(phi_dot, 1, 1, 1)

    E_x_dot = outgoing_wave(maxwell, E_x_dot, E_x)
    E_y_dot = outgoing_wave(maxwell, E_y_dot, E_y)
    E_z_dot = outgoing_wave(maxwell, E_z_dot, E_z)

    A_x_dot = outgoing_wave(maxwell, A_x_dot, A_x)
    A_y_dot = outgoing_wave(maxwell, A_y_dot, A_y)
    A_z_dot = outgoing_wave(maxwell, A_z_dot, A_z)

    phi_dot = outgoing_wave(maxwell, phi_dot, phi)

    return (E_x_dot, E_y_dot, E_z_dot, A_x_dot, A_y_dot, A_z_dot, phi_dot)


# Background: In numerical simulations, updating fields over time is a fundamental step in solving differential equations.
# This process involves using the time derivatives of the fields to compute their new values at the next time step.
# The update is typically performed using a time integration scheme, such as the Euler method or higher-order methods
# like Runge-Kutta. The update formula generally involves adding a scaled increment of the time derivative to the
# current field value. The scaling factor and time step size determine the magnitude of the update. This approach
# allows the simulation to evolve the fields over time, capturing the dynamics described by the underlying equations.

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
    new_fields = []
    for field, field_dot in zip(fields, fields_dot):
        # Update each field by adding the scaled time derivative
        updated_field = field + factor * dt * field_dot
        new_fields.append(updated_field)
    
    return new_fields


# Background: The Crank-Nicholson method is a numerical technique used to solve differential equations, particularly
# partial differential equations, by providing a stable and accurate time integration scheme. It is an implicit method
# that averages the results of the forward Euler method and the backward Euler method, making it second-order accurate
# in time. In the context of electromagnetic simulations, the Crank-Nicholson method can be used to update the fields
# over time while maintaining stability. The Courant number is a dimensionless number that ensures stability in numerical
# solutions of hyperbolic partial differential equations. It is used to control the time step size relative to the spatial
# grid size and the wave speed. In this implementation, the Crank-Nicholson method is iteratively applied using a Runge-Kutta
# scheme to integrate the fields from the current time to a specified future time, with the substep size controlled by the
# Courant number to ensure stability.


def crank_nicholson_step(maxwell, fields, dt):
    '''Performs a single Crank-Nicholson step to update the fields.
    Parameters:
    -----------
    maxwell : object
        An object representing the Maxwell simulation environment.
    fields : list of numpy.ndarray
        A list containing the current field values to be updated.
    dt : float
        Time step size for the Crank-Nicholson step.
    Returns:
    --------
    list of numpy.ndarray
        A list containing the updated fields after the Crank-Nicholson step.
    '''
    # Compute initial time derivatives
    fields_dot = derivatives(maxwell, fields)
    
    # First half-step
    half_step_fields = update_fields(maxwell, fields, fields_dot, 0.5, dt)
    
    # Compute time derivatives at the half-step
    half_step_fields_dot = derivatives(maxwell, half_step_fields)
    
    # Full step using the average of initial and half-step derivatives
    new_fields = []
    for field, field_dot, half_field_dot in zip(fields, fields_dot, half_step_fields_dot):
        updated_field = field + dt * (0.5 * (field_dot + half_field_dot))
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
    # Calculate the substep size using the Courant number
    delta_t = courant * maxwell.delta / sqrt(3)  # sqrt(3) for 3D stability
    num_steps = int(np.ceil(t_const / delta_t))
    delta_t = t_const / num_steps  # Adjust delta_t to fit exactly into t_const

    # Perform the iterative Crank-Nicholson steps
    for _ in range(num_steps):
        fields = crank_nicholson_step(maxwell, fields, delta_t)
        maxwell.t += delta_t

    return maxwell.t, fields



# Background: In the context of the 3+1 formulation of Maxwell's equations, the constraint equation ensures that the divergence
# of the electric field E is balanced by the charge density ρ. In a source-free evolution, where the charge density ρ is zero,
# the constraint simplifies to D_i E^i = 0, meaning the divergence of the electric field should be zero at each point in space.
# This constraint must be satisfied at each time step to ensure the physical accuracy of the simulation. The constraint violation
# can be quantified by calculating the divergence of the electric field and checking how much it deviates from zero. The L2 norm
# of the constraint violation provides a measure of the overall deviation across the entire grid, which can be used to monitor
# the accuracy and stability of the simulation.

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
    # Extract electric field components and grid spacing
    E_x, E_y, E_z = maxwell.E_x, maxwell.E_y, maxwell.E_z
    delta = maxwell.delta

    # Compute the divergence of the electric field
    div_E = np.zeros_like(E_x)
    nx, ny, nz = E_x.shape

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                div_E[i, j, k] = (
                    (E_x[i + 1, j, k] - E_x[i - 1, j, k]) / (2 * delta) +
                    (E_y[i, j + 1, k] - E_y[i, j - 1, k]) / (2 * delta) +
                    (E_z[i, j, k + 1] - E_z[i, j, k - 1]) / (2 * delta)
                )

    # Calculate the L2 norm of the constraint violation
    norm_c = np.sqrt(np.sum(div_E**2) * (delta**3))

    return norm_c


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('13.12', 3)
target = targets[0]

maxwell = Maxwell(50,2)
x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
fields = (x,y,z,x,y,z, z*0+1)
courant  = 0.5
t_const  = 0.2
stepper(maxwell, fields, courant, t_const)
assert np.allclose(check_constraint(maxwell), target)
target = targets[1]

maxwell = Maxwell(50,2)
x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
fields = (x,y,z,-y,x,z*0, z*0+1)
courant  = 0.5
t_const  = 0.2
stepper(maxwell, fields, courant, t_const)
assert np.allclose(check_constraint(maxwell), target)
target = targets[2]

maxwell = Maxwell(50,2)
x,y,z = np.meshgrid(*[np.linspace(0,2,50)]*3)
fields = (x,y,z,x*0,-z,y, z*0+1)
courant  = 0.5
t_const  = 0.2
stepper(maxwell, fields, courant, t_const)
assert np.allclose(check_constraint(maxwell), target)
