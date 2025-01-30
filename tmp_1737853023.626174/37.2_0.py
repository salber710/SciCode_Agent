import numpy as np

# Background: 
# In optics, paraxial rays are those that make small angles with the optical axis of a lens system. 
# The paraxial approximation simplifies the analysis of optical systems by assuming that sin(θ) ≈ θ and tan(θ) ≈ θ for small angles θ.
# This approximation is useful for calculating the path of light rays through lenses, especially in systems with spherical surfaces.
# The doublet lens system consists of two lenses with different curvatures and refractive indices, separated by a distance.
# The goal is to calculate the intersection of paraxial rays with the optical axis after passing through the lens system.
# The lensmaker's equation and Snell's law are used to determine the refraction at each lens surface.
# The position of the third lens is used as the origin for calculating the axial image location.


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
    
    # Validate inputs
    if any(np.array([r1, r2, r3]) == 0):
        raise ValueError("Radius of curvature cannot be zero.")
    if any(np.array([n1, n2, n3, n_total]) <= 0):
        raise ValueError("Refractive indices must be positive.")
    if d1 < 0 or d2 < 0:
        raise ValueError("Separation distances must be non-negative.")
    
    # Calculate the focal lengths using the lensmaker's equation for each lens
    f1 = 1 / ((n1 - n_total) * (1/r1 - 1/r2 + (n1 - n_total) * d1 / (n1 * r1 * r2)))
    f2 = 1 / ((n2 - n1) * (1/r2 - 1/r3 + (n2 - n1) * d2 / (n2 * r2 * r3)))
    
    # Calculate the effective focal length of the doublet lens system
    F = 1 / (1/f1 + 1/f2 - d1/(f1*f2))
    
    # Calculate the image location for each incident height using the paraxial approximation
    l31 = F * (1 - (h1**2 / (2 * F**2)))
    
    # Adjust the calculation for zero aperture height to return zero
    l31[h1 == 0] = 0
    
    return l31



# Background: 
# In optics, non-paraxial rays are those that do not adhere to the small angle approximation, meaning they can make larger angles with the optical axis.
# This requires a more complex analysis as the approximations sin(θ) ≈ θ and tan(θ) ≈ θ are not valid.
# For non-paraxial rays, Snell's law must be applied directly to calculate the refraction at each lens surface.
# The lensmaker's equation is still used to determine the focal lengths, but the ray tracing must account for the actual angles.
# The goal is to calculate the intersection of non-paraxial rays with the optical axis after passing through the lens system.
# The position of the third lens is used as the origin for calculating the axial image location.


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
    
    # Validate inputs
    if any(np.array([r1, r2, r3]) == 0):
        raise ValueError("Radius of curvature cannot be zero.")
    if any(np.array([n1, n2, n3, n_total]) <= 0):
        raise ValueError("Refractive indices must be positive.")
    if d1 < 0 or d2 < 0:
        raise ValueError("Separation distances must be non-negative.")
    
    # Calculate the focal lengths using the lensmaker's equation for each lens
    f1 = 1 / ((n1 - n_total) * (1/r1 - 1/r2 + (n1 - n_total) * d1 / (n1 * r1 * r2)))
    f2 = 1 / ((n2 - n1) * (1/r2 - 1/r3 + (n2 - n1) * d2 / (n2 * r2 * r3)))
    
    # Calculate the effective focal length of the doublet lens system
    F = 1 / (1/f1 + 1/f2 - d1/(f1*f2))
    
    # Initialize the array for non-paraxial image locations
    L31 = np.zeros_like(h1)
    
    # Calculate the image location for each incident height using Snell's law
    for i, h in enumerate(h1):
        # Calculate the angle of incidence at the first surface
        theta1 = np.arcsin(h / r1)
        
        # Refraction at the first surface
        theta2 = np.arcsin((n_total / n1) * np.sin(theta1))
        
        # Calculate the height at the second surface
        h2 = r2 * np.sin(theta2)
        
        # Refraction at the second surface
        theta3 = np.arcsin((n1 / n2) * np.sin(theta2))
        
        # Calculate the height at the third surface
        h3 = r3 * np.sin(theta3)
        
        # Refraction at the third surface
        theta4 = np.arcsin((n2 / n3) * np.sin(theta3))
        
        # Calculate the final image location using the angle at the third surface
        L31[i] = h3 / np.tan(theta4)
    
    return L31

from scicode.parse.parse import process_hdf5_to_tuple
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
