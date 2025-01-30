from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def calculate_paraxial(h1, r1, r2, r3, d1, d2, n1, n2, n3, n_total):
    # Calculate the optical power of each surface using a different approach
    P1 = (n1 - n_total) / r1
    P2 = (n2 - n1) / r2
    P3 = (n3 - n2) / r3
    
    # Calculate the net power considering the distances and indices with a different approach
    P_net = P1 + P2 + P3 - (P1 * P2 * d1 / n1) - (P2 * P3 * d2 / n2)
    
    # Effective focal length of the system
    F_eff = 1 / P_net
    
    # Calculate the image position using a hyperbolic approximation for small angles
    l31 = F_eff * np.cosh(h1 / F_eff)
    
    return l31



# Background: In optical systems, non-paraxial rays do not adhere to the small angle approximation, which is valid for paraxial rays. 
# Non-paraxial rays can have significant angles with respect to the optical axis, leading to more complex behavior such as spherical aberration.
# To calculate the intersection of non-paraxial rays with the optical axis, we need to consider the exact refraction at each lens surface using Snell's Law.
# This involves calculating the angle of incidence and refraction at each surface and tracing the ray through the lens system.
# The position of the third lens is used as the origin for these calculations.


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
    
    # Initialize an array to store the intersection positions
    L31 = np.zeros_like(h1)
    
    # Iterate over each incident height
    for i, h in enumerate(h1):
        # Calculate the angle of incidence at the first surface
        theta1 = np.arcsin(h / r1)
        
        # Calculate the angle of refraction using Snell's Law
        theta2 = np.arcsin((n_total / n1) * np.sin(theta1))
        
        # Calculate the height at the second surface
        h2 = h + d1 * np.tan(theta2)
        
        # Calculate the angle of incidence at the second surface
        theta3 = np.arcsin(h2 / r2)
        
        # Calculate the angle of refraction at the second surface
        theta4 = np.arcsin((n1 / n2) * np.sin(theta3))
        
        # Calculate the height at the third surface
        h3 = h2 + d2 * np.tan(theta4)
        
        # Calculate the angle of incidence at the third surface
        theta5 = np.arcsin(h3 / r3)
        
        # Calculate the angle of refraction at the third surface
        theta6 = np.arcsin((n2 / n3) * np.sin(theta5))
        
        # Calculate the intersection position on the optical axis
        L31[i] = h3 / np.tan(theta6)
    
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