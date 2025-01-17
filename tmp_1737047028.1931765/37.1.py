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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('37.1', 3)
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
assert np.allclose(calculate_paraxial(h1, r1, r2, r3, d1, d2, nD1, nD1, nD3, n), target)
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
assert np.allclose(calculate_paraxial(h1, r1, r2, r3, d1, d2, nF1, nF1, nF3, n), target)
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
assert np.allclose(calculate_paraxial(h1, r1, r2, r3, d1, d2, nC1, nC1, nC3, n), target)
