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
# In optics, non-paraxial rays are those that make larger angles with the optical axis, 
# for which the paraxial approximations (sin(θ) ≈ θ, tan(θ) ≈ θ) do not hold. 
# When dealing with non-paraxial rays, one must use the full trigonometric functions 
# to accurately model the refraction and propagation of light through a lens system.
# For a doublet lens system, the challenge is to compute the intersection of these rays 
# with the optical axis after passing through the lens surfaces. This involves calculating 
# the refraction at each surface using Snell's Law, which relates the angle of incidence 
# and refraction to the refractive indices of the media. The ray tracing approach requires 
# iterative calculations through each interface, taking into account the curvature of the 
# lens surfaces, the separation distances, and the refractive indices of the lens materials.


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
    
    # Initialize list to store the intersection points
    L31 = []
    
    # Iterate over each incident height to calculate the non-paraxial intersection
    for h in h1:
        # Calculate the angle of incidence for the first surface
        theta1 = np.arcsin(h / r1)
        
        # Refraction at the first surface using Snell's Law
        theta2 = np.arcsin((n_total / n1) * np.sin(theta1))
        
        # Calculate the angle of incidence for the second surface
        # Adjusting for the path through the first lens
        h2 = r1 * np.tan(theta2)
        theta3 = np.arcsin(h2 / r2)
        
        # Refraction at the second surface
        theta4 = np.arcsin((n1 / n2) * np.sin(theta3))
        
        # Calculate the angle of incidence for the third surface
        # Adjusting for the path through the second lens
        h3 = r2 * np.tan(theta4)
        theta5 = np.arcsin(h3 / r3)
        
        # Refraction at the third surface
        theta6 = np.arcsin((n2 / n3) * np.sin(theta5))
        
        # Calculate the final intersection point on the optical axis
        # Assuming the third surface position as the origin
        L3 = r3 * np.tan(theta6)
        
        # Append the result to the list
        L31.append(L3)
    
    return np.array(L31)

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
