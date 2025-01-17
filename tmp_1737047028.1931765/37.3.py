import numpy as np

# Background: 
# In optics, paraxial rays are those rays that make small angles with the optical axis of a lens system. 
# The paraxial approximation simplifies the analysis of optical systems by assuming that sin(θ) ≈ θ and tan(θ) ≈ θ, 
# where θ is the angle the ray makes with the optical axis. This approximation is valid for small angles and is 
# used to derive the lens maker's equation and other fundamental optical formulas.
# In a doublet lens system, which consists of two lenses, the curvature of each lens surface and the refractive 
# indices of the materials determine how light is refracted through the system. The goal is to calculate the 
# intersection of paraxial rays with the optical axis after passing through the lens system, which is crucial 
# for understanding image formation and spherical aberration.
# The calculation involves using the lens maker's formula and the refraction at each surface to determine the 
# path of the rays. The position of the third lens is used as the origin for these calculations.


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
    
    # Calculate the focal lengths using the lens maker's formula for each lens
    f1 = 1 / ((n1 - n_total) * (1/r1 - 1/r2 + (n1 - n_total) * d1 / (n1 * r1 * r2)))
    f2 = 1 / ((n2 - n1) * (1/r2 - 1/r3 + (n2 - n1) * d2 / (n2 * r2 * r3)))
    
    # Calculate the effective focal length of the doublet lens system
    F = 1 / (1/f1 + 1/f2 - d1/(f1*f2))
    
    # Calculate the image position for each incident height using the lens formula
    l31 = F * (1 - (h1**2 / (2 * F**2)))
    
    return l31


# Background: In optics, non-paraxial rays are those that do not adhere to the small angle approximation, meaning they can make larger angles with the optical axis. This requires a more complex analysis as the approximations sin(θ) ≈ θ and tan(θ) ≈ θ are no longer valid. Instead, the exact trigonometric functions must be used to calculate the refraction and path of the rays through the lens system. This is crucial for accurately modeling the behavior of light in systems where rays are not close to the optical axis, such as in wide-angle lenses or when considering spherical aberration in detail. The Snell's law of refraction is used to determine the angle of refraction at each lens surface, and the geometry of the lens system is used to trace the path of the rays to find their intersection with the optical axis.


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
    
    # Initialize an array to store the image locations
    L31 = np.zeros_like(h1)
    
    # Iterate over each incident height
    for i, h in enumerate(h1):
        # Calculate the angle of incidence at the first surface
        theta1 = np.arcsin(h / r1)
        
        # Calculate the angle of refraction at the first surface using Snell's law
        theta2 = np.arcsin((n_total / n1) * np.sin(theta1))
        
        # Calculate the path length to the second surface
        path_length1 = d1 / np.cos(theta2)
        
        # Calculate the angle of incidence at the second surface
        theta3 = np.arcsin((h + path_length1 * np.tan(theta2)) / r2)
        
        # Calculate the angle of refraction at the second surface
        theta4 = np.arcsin((n1 / n2) * np.sin(theta3))
        
        # Calculate the path length to the third surface
        path_length2 = d2 / np.cos(theta4)
        
        # Calculate the angle of incidence at the third surface
        theta5 = np.arcsin((h + path_length1 * np.tan(theta2) + path_length2 * np.tan(theta4)) / r3)
        
        # Calculate the angle of refraction at the third surface
        theta6 = np.arcsin((n2 / n3) * np.sin(theta5))
        
        # Calculate the intersection with the optical axis
        L31[i] = (h + path_length1 * np.tan(theta2) + path_length2 * np.tan(theta4)) / np.tan(theta6)
    
    return L31



# Background: Spherical aberration is an optical effect observed in lenses where rays that strike the lens near its edge are focused at different points compared to rays that strike near the center. This occurs because the lens surfaces are spherical, leading to variations in focal length for different incident heights. In a doublet lens system, spherical aberration can be analyzed by comparing the image positions calculated using paraxial (small angle approximation) and non-paraxial (exact trigonometric functions) methods. The difference between these positions for various incident heights gives a measure of the spherical aberration. This is crucial for understanding and correcting image distortions in optical systems.


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
    
    # Calculate paraxial image locations
    f1 = 1 / ((n1 - n_total) * (1/r1 - 1/r2 + (n1 - n_total) * d1 / (n1 * r1 * r2)))
    f2 = 1 / ((n2 - n1) * (1/r2 - 1/r3 + (n2 - n1) * d2 / (n2 * r2 * r3)))
    F = 1 / (1/f1 + 1/f2 - d1/(f1*f2))
    l31_paraxial = F * (1 - (h1**2 / (2 * F**2)))
    
    # Calculate non-paraxial image locations
    L31_non_paraxial = np.zeros_like(h1)
    for i, h in enumerate(h1):
        theta1 = np.arcsin(h / r1)
        theta2 = np.arcsin((n_total / n1) * np.sin(theta1))
        path_length1 = d1 / np.cos(theta2)
        theta3 = np.arcsin((h + path_length1 * np.tan(theta2)) / r2)
        theta4 = np.arcsin((n1 / n2) * np.sin(theta3))
        path_length2 = d2 / np.cos(theta4)
        theta5 = np.arcsin((h + path_length1 * np.tan(theta2) + path_length2 * np.tan(theta4)) / r3)
        theta6 = np.arcsin((n2 / n3) * np.sin(theta5))
        L31_non_paraxial[i] = (h + path_length1 * np.tan(theta2) + path_length2 * np.tan(theta4)) / np.tan(theta6)
    
    # Calculate spherical aberration as the difference between non-paraxial and paraxial image locations
    LC = L31_non_paraxial - l31_paraxial
    
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
