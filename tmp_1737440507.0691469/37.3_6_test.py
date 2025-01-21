import numpy as np

# Background: 
# In optics, the paraxial approximation is used for rays that make small angles with the optical axis, 
# allowing us to simplify calculations by assuming that sine and tangent of the angle are approximately equal to the angle itself (in radians).
# For a doublet lens system, which consists of two lenses cemented together, the calculation of the intersection of paraxial rays with the optical axis 
# involves tracing the path of light through each lens surface and calculating refractions using Snell's law.
# The refraction at each surface can be calculated using the lensmaker's equation for paraxial rays:
#   (n2/n1 - 1) * (1/r1 - 1/r2) for the first and second surfaces.
# The position of intersection of paraxial rays with the optical axis is determined by the lens power, 
# separation distances, and the refractive indices of the lens materials.
# The paraxial focus position for each surface is calculated iteratively by considering the changes in direction due to refraction at each surface.
# Given parameters include the radii of curvature (r1, r2, r3), the distances between surfaces (d1, d2), 
# and the refractive indices of the materials (n1, n2, n3) and surroundings (n_total).


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

    # Calculate the focal length of the first lens using the lensmaker's equation
    f1 = 1 / ((n1 - n_total) * (1/r1 - 1/r2))
    
    # Calculate the focal length of the second lens
    f2 = 1 / ((n2 - n1) * (1/r2 - 1/r3))
    
    # Calculate the effective focal length of the doublet lens (two thin lenses in contact)
    f_eff = 1 / ((1/f1) + (1/f2) - (d1/(f1*f2)))
    
    # Calculate the position of the axial image for each height h1 using paraxial approximation
    l31 = h1 * f_eff / (h1 + f_eff)
    
    return l31


# Background:
# In optical systems, when dealing with non-paraxial rays, we cannot apply the small angle approximation that simplifies the trigonometric 
# functions to their angles. Instead, we must consider the full trigonometric functions when calculating refraction and ray paths.
# For non-paraxial rays, we use Snell's law to calculate the exact angles of refraction at each lens surface. The path of each ray is 
# traced through the optical system by computing the intersection points with each surface and applying Snell's law to determine 
# the new direction of the ray.
# The refraction is calculated using Snell's law: n1 * sin(theta1) = n2 * sin(theta2), where theta1 is the incident angle and theta2 
# is the refracted angle. We must calculate these angles without approximation to find the intersection positions accurately.
# The goal is to track how these rays intersect with the optical axis after passing through the doublet lens.


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

    # Initialize array for L31
    L31 = np.zeros_like(h1)
    
    # Iterate over each incident height
    for i, h in enumerate(h1):
        # Calculate incidence angle for the first lens surface
        theta1 = np.arcsin(h / r1)  # Approximate initial angle using geometry
        
        # Snell's Law: n_total * sin(theta1) = n1 * sin(theta2)
        # Calculate refraction at the first surface
        sin_theta2 = n_total * np.sin(theta1) / n1
        theta2 = np.arcsin(sin_theta2)
        
        # Calculate the path to the second surface
        x1 = r1 - np.cos(theta2) * (r1 - h / np.tan(theta2))
        y1 = h + np.sin(theta2) * (r1 - h / np.tan(theta2))
        
        # Calculate incidence angle for the second lens surface
        theta3 = np.arcsin((y1 - d1) / r2)  # Approximate angle
        
        # Snell's Law: n1 * sin(theta3) = n2 * sin(theta4)
        sin_theta4 = n1 * np.sin(theta3) / n2
        theta4 = np.arcsin(sin_theta4)
        
        # Calculate the path to the third surface
        x2 = x1 + np.cos(theta4) * d1
        y2 = y1 + np.sin(theta4) * d1
        
        # Calculate incidence angle for the third lens surface
        theta5 = np.arcsin((y2 - d2) / r3)  # Approximate angle
        
        # Snell's Law: n2 * sin(theta5) = n3 * sin(theta6)
        sin_theta6 = n2 * np.sin(theta5) / n3
        theta6 = np.arcsin(sin_theta6)
        
        # Calculate the intersection with the optical axis
        L31[i] = x2 + (y2 / np.tan(theta6))
    
    return L31



# Background: 
# Spherical aberration occurs when light rays passing through a lens do not converge at a single focal point, 
# causing the image to be blurred. This is primarily due to the variation in focal length for rays passing at different 
# distances from the optical axis. In optical systems, spherical aberration is analyzed by comparing the intersection 
# positions of paraxial (idealized small angle) and non-paraxial (realistic full angle) rays with the optical axis.
# For a doublet lens, we compute spherical aberration by taking the difference between the intersection points of 
# paraxial and non-paraxial rays. This difference, known as longitudinal spherical aberration, varies with the 
# incident height on the lens. The goal is to quantify this aberration and understand its dependence on the incident height.


def compute_LC(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    '''Computes spherical aberration by comparing paraxial and axial calculations.
    Parameters:
    - h1 (array of floats): Aperture heights, in range (0.01, hm)
    - r1, r2, r3 (floats): Radii of curvature of the three surfaces
    - d1, d2 (floats): Separation distances between surfaces
    - n1, n2, n3 (floats): Refractive indices for the three lens materials
    - n_total (float): Refractive index of the surrounding medium (air)
    Returns:
    - LC (array of floats): Spherical aberration (difference between paraxial and axial)
    '''
    
    # Calculate paraxial intersection points
    paraxial_l31 = calculate_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total)

    # Calculate non-paraxial intersection points
    non_paraxial_L31 = calculate_non_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total)

    # Compute the spherical aberration as the difference between non-paraxial and paraxial intersections
    LC = non_paraxial_L31 - paraxial_l31

    return LC

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('37.3', 4)
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
assert np.allclose(compute_LC(h1, r1, r2, r3, d1, d2, nD1, nD1, nD3, n), target)
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
assert np.allclose(compute_LC(h1, r1, r2, r3, d1, d2, nF1, nF1, nF3, n), target)
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
assert np.allclose(compute_LC(h1, r1, r2, r3, d1, d2, nC1, nC1, nC3, n), target)
target = targets[3]

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
LCC= compute_LC(h1, r1, r2, r3, d1, d2, nC1, nC1, nC3, n)
assert (LCC[0]<LCC[-1]) == target
