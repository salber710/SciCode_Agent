from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy



def Rlnm(l, n, m, k, z, N_t):
    '''Function to calculate the translation coefficient (R|R)lmn using a hybrid approach with Chebyshev polynomials.
    Input
    l : int
        The principal quantum number for the expansion.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    k : float
        Wavevector of the optical beam.
    z : float
        The translation distance along z direction
    N_t : int
        Truncated space size.
    Output
    (R|R)lmn : complex
        The translation coefficient (R|R)lmn.
    '''
    # Initialize the translation coefficient
    translation_coefficient = 0.0 + 0.0j

    # Calculate the argument for the spherical Bessel functions
    kr = k * z

    # Use Chebyshev polynomials to approximate the Legendre polynomial calculations
    for l_prime in range(N_t):
        # Calculate the spherical Bessel function of the first kind
        j_l = scipy.special.spherical_jn(l_prime, kr)

        # Calculate the Chebyshev polynomial of the first kind
        T_lm = np.polynomial.chebyshev.Chebyshev.basis(l_prime)(np.cos(z))

        # Update the translation coefficient using the Chebyshev polynomial approximation
        translation_coefficient += (2 * l_prime + 1) * j_l * T_lm

    # Normalize the translation coefficient
    translation_coefficient *= (4 * np.pi) / (2 * l + 1)

    return translation_coefficient




def Tnvm(n, v, m, Q):
    # Define the transformation from Cartesian to spherical coordinates
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    # Initialize the rotation coefficient
    T = 0.0 + 0.0j

    # Define integration grid over the sphere
    phi = np.linspace(0, 2 * np.pi, 200)
    theta = np.linspace(0, np.pi, 100)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]
    sin_theta = np.sin(theta_grid)

    # Compute spherical harmonics for the original coordinates
    Ynv = sph_harm(v, n, phi_grid, theta_grid)

    # Compute spherical harmonics for the rotated coordinates
    for i in range(len(theta)):
        for j in range(len(phi)):
            # Original coordinates
            x = np.sin(theta_grid[i, j]) * np.cos(phi_grid[i, j])
            y = np.sin(theta_grid[i, j]) * np.sin(phi_grid[i, j])
            z = np.cos(theta_grid[i, j])

            # Apply rotation
            rotated_vec = Q @ np.array([x, y, z])
            r_rot, theta_rot, phi_rot = cartesian_to_spherical(*rotated_vec)

            # Compute spherical harmonics at the new location
            Y_rot = sph_harm(m, n, phi_rot, theta_rot)

            # Integrate using the composite trapezoidal rule
            T += np.conj(Y_rot) * Ynv[i, j] * sin_theta[i, j] * dtheta * dphi

    return T



# Background: In this step, we need to calculate the reexpansion coefficient BR_nm for an optical beam that undergoes an arbitrary translation. 
# The translation is decomposed into a sequence of operations: a rotation to align the z-axis with the translation vector r0, a translation along the new z-axis, 
# and a rotation back to the original coordinate system. The reexpansion coefficient is computed by combining the effects of these transformations on the spherical harmonics 
# and the expansion coefficients of the elementary regular solutions. The process involves using rotation matrices and translation coefficients calculated in previous steps.

def compute_BRnm(r0, B, n, m, wl, N_t):
    '''Function to calculate the reexpansion coefficient BR_nm.
    Input
    r_0 : array
        Translation vector.
    B : matrix of shape(N_t + 1, 2 * N_t + 1)
        Expansion coefficients of the elementary regular solutions.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    wl : float
        Wavelength of the optical beam.
    N_t : int
        Truncated space size.
    Output
    BR_nm : complex
        Reexpansion coefficient BR_nm of the elementary regular solutions.
    '''
    # Calculate the magnitude of the translation vector
    r0_magnitude = np.linalg.norm(r0)
    
    # Calculate the wavevector
    k = 2 * np.pi / wl
    
    # Calculate the rotation matrix to align z-axis with r0
    z_axis = np.array([0, 0, 1])
    r0_unit = r0 / r0_magnitude
    v = np.cross(z_axis, r0_unit)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, r0_unit)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2)) if s != 0 else np.eye(3)
    
    # Initialize the reexpansion coefficient
    BRnm = 0.0 + 0.0j
    
    # Iterate over the expansion coefficients
    for l in range(N_t + 1):
        for mu in range(-l, l + 1):
            # Calculate the rotation coefficient T_n^{mu m}
            Tnvm_value = Tnvm(n, mu, m, R)
            
            # Calculate the translation coefficient (R|R)_{ln}^{mu}
            Rlnm_value = Rlnm(l, n, mu, k, r0_magnitude, N_t)
            
            # Update the reexpansion coefficient
            BRnm += B[l, mu + N_t] * Tnvm_value * Rlnm_value
    
    return BRnm


try:
    targets = process_hdf5_to_tuple('22.3', 4)
    target = targets[0]
    r0 = np.array([0.5, 0, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t] = 1
    wl = 2 * np.pi
    n = 2
    m = 1
    assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)

    target = targets[1]
    r0 = np.array([0.5, 0.5, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t] = 1
    wl = 2 * np.pi
    n = 2
    m = 1
    assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)

    target = targets[2]
    r0 = np.array([0.5, 1, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t] = 1
    wl = 2 * np.pi
    n = 2
    m = 1
    assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)

    target = targets[3]
    r0 = np.array([0, 0.5, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t + 1] = 1
    wl = 2 * np.pi
    n = 2
    m = 2
    assert (compute_BRnm(r0, B, n, m, wl, N_t) == 0) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e