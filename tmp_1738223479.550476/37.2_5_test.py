from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def calculate_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    '''Computes the axial parameters for spherical aberration calculation.
    Parameters:
    - h1 (array of floats): Aperture heights, in range (0.01, hm)
    - r1 (float): Radius of curvature for the first lens surface
    - r2 (float): Radius of curvature for the second lens surface
    - r3 (float): Radius of curvature for the third lens surface
    - d1 (float): Separation distance between first and second surfaces
    - d2 (float): Separation distance between second and third surfaces
    - n1 (float): Refractive index of the first lens material
    - n2 (float): Refractive index of the second lens material
    - n3 (float): Refractive index of the third lens material
    - n_total (float): Refractive index of the surrounding medium (air)
    Returns:
    - l31 (array of floats): Axial image locations for the third surface
    '''


    # Implement a new method using a different conceptual approach: ray transfer matrix method
    def transfer_matrix(n1, n2, r):
        """Return the transfer matrix for a single spherical surface."""
        return np.array([[1, 0], [(n2 - n1) / r, n1 / n2]])

    def propagation_matrix(d, n):
        """Return the transfer matrix for free space propagation."""
        return np.array([[1, d / n], [0, 1]])

    # Initialize the ray as a column vector [height; angle]
    rays = np.array([h1, np.zeros_like(h1)])

    # Calculate transfer matrices for each interface
    M1 = transfer_matrix(n_total, n1, r1)
    M2 = transfer_matrix(n1, n2, r2)
    M3 = transfer_matrix(n2, n3, r3)

    # Calculate propagation matrices for the distances between lenses
    P1 = propagation_matrix(d1, n1)
    P2 = propagation_matrix(d2, n2)

    # Sequentially apply the matrices to get the final ray parameters at the third surface
    rays = np.dot(M3, np.dot(P2, np.dot(M2, np.dot(P1, np.dot(M1, rays)))))

    # The final ray height on the third surface is the axial image location
    l31 = rays[0]

    return l31




def calculate_non_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    '''Computes the non-paraxial parameters for spherical aberration calculation.
    Parameters:
    - h1 (array of floats): Aperture heights, in range (0.01, hm)
    - r1 (float): Radius of curvature for the first lens surface
    - r2 (float): Radius of curvature for the second lens surface
    - r3 (float): Radius of curvature for the third lens surface
    - d1 (float): Separation distance between first and second surfaces
    - d2 (float): Separation distance between second and third surfaces
    - n1 (float): Refractive index of the first lens material
    - n2 (float): Refractive index of the second lens material
    - n3 (float): Refractive index of the third lens material
    - n_total (float): Refractive index of the surrounding medium (air)
    Returns:
    - L31 (array of floats): Non-paraxial image locations for the third surface
    '''

    # Calculate initial angles using a hybrid approach: combining arcsin and arccos
    angles = np.arctan2(h1, r1)
    rays = np.vstack((h1, angles))
    
    def snell_refraction(n1, n2, angle_incident):
        """Calculate the angle of refraction using Snell's law."""
        return np.arctan2(n1 * np.sin(angle_incident), n2)
    
    def refract_at_surface(n1, n2, r, rays):
        """Calculate the refraction at a spherical surface."""
        h, angle_incident = rays
        angle_refracted = snell_refraction(n1, n2, angle_incident)
        adjusted_angle = angle_refracted - np.arctan2(h, r)
        return np.vstack((h, adjusted_angle))
    
    def propagate_through_distance(d, rays):
        """Propagate the ray through a distance."""
        h, angle = rays
        new_height = h + d * np.arcsinh(np.tan(angle))  # Using hyperbolic sine for propagation
        return np.vstack((new_height, angle))
    
    # Sequential refraction and propagation
    rays = refract_at_surface(n_total, n1, r1, rays)
    rays = propagate_through_distance(d1, rays)
    rays = refract_at_surface(n1, n2, r2, rays)
    rays = propagate_through_distance(d2, rays)
    rays = refract_at_surface(n2, n3, r3, rays)
    
    # Compute the final image positions on the third surface
    L31 = rays[0]

    return L31


try:
    targets = process_hdf5_to_tuple('37.2', 3)
    target = targets[0]
    n = 1.000       # (float) Refractive index of air
    nD1 = 1.51470   # (float) Refractive index of the first lens material for D light
    nD3 = 1.67270   # (float) Refractive index of the third lens material for D light
    nF1 = 1.52067   # (float) Refractive index of the first lens material for F light
    nF3 = 1.68749   # (float) Refractive index of the third lens material for F light
    nC1 = 1.51218   # (float) Refractive index of the first lens material for C light
    nC3 = 1.66662   # (float) Refractive index of the third lens material for C light
    r1 = 61.857189  # (float) Radius of curvature of the first lens surface
    r2 = -43.831719 # (float) Radius of curvature of the second lens surface
    r3 = -128.831547 # (float) Radius of curvature of the third lens surface
    d1 = 1.9433     # (float) Separation distance between first and second surfaces (lens thickness)
    d2 = 1.1        # (float) Separation distance between second and third surfaces
    hm = 20         # (float) Maximum aperture height
    h1 = np.linspace(0.01, hm, 1000)
    assert np.allclose(calculate_non_paraxial(h1, r1, r2, r3, d1, d2, nD1, nD1, nD3, n), target)

    target = targets[1]
    n = 1.000       # (float) Refractive index of air
    nD1 = 1.51470   # (float) Refractive index of the first lens material for D light
    nD3 = 1.67270   # (float) Refractive index of the third lens material for D light
    nF1 = 1.52067   # (float) Refractive index of the first lens material for F light
    nF3 = 1.68749   # (float) Refractive index of the third lens material for F light
    nC1 = 1.51218   # (float) Refractive index of the first lens material for C light
    nC3 = 1.66662   # (float) Refractive index of the third lens material for C light
    r1 = 61.857189  # (float) Radius of curvature of the first lens surface
    r2 = -43.831719 # (float) Radius of curvature of the second lens surface
    r3 = -128.831547 # (float) Radius of curvature of the third lens surface
    d1 = 1.9433     # (float) Separation distance between first and second surfaces (lens thickness)
    d2 = 1.1        # (float) Separation distance between second and third surfaces
    hm = 20         # (float) Maximum aperture height
    h1 = np.linspace(0.01, hm, 1000)
    assert np.allclose(calculate_non_paraxial(h1, r1, r2, r3, d1, d2, nF1, nF1, nF3, n), target)

    target = targets[2]
    n = 1.000       # (float) Refractive index of air
    nD1 = 1.51470   # (float) Refractive index of the first lens material for D light
    nD3 = 1.67270   # (float) Refractive index of the third lens material for D light
    nF1 = 1.52067   # (float) Refractive index of the first lens material for F light
    nF3 = 1.68749   # (float) Refractive index of the third lens material for F light
    nC1 = 1.51218   # (float) Refractive index of the first lens material for C light
    nC3 = 1.66662   # (float) Refractive index of the third lens material for C light
    r1 = 61.857189  # (float) Radius of curvature of the first lens surface
    r2 = -43.831719 # (float) Radius of curvature of the second lens surface
    r3 = -128.831547 # (float) Radius of curvature of the third lens surface
    d1 = 1.9433     # (float) Separation distance between first and second surfaces (lens thickness)
    d2 = 1.1        # (float) Separation distance between second and third surfaces
    hm = 20         # (float) Maximum aperture height
    h1 = np.linspace(0.01, hm, 1000)
    assert np.allclose(calculate_non_paraxial(h1, r1, r2, r3, d1, d2, nC1, nC1, nC3, n), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e