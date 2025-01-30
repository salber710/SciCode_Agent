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


    # Define function to calculate surface power
    def surface_power(n1, n2, r):
        return (n2 - n1) / r if r != 0 else 0
    
    # Surface powers for each lens surface
    power1 = surface_power(n_total, n1, r1)
    power2 = surface_power(n1, n2, r2)
    power3 = surface_power(n2, n3, r3)

    # Calculate the shift in focus due to each surface
    d_eff1 = (n1 / power1) if power1 != 0 else np.inf
    d_eff2 = (n2 / power2) if power2 != 0 else np.inf
    d_eff3 = (n3 / power3) if power3 != 0 else np.inf

    # Using effective distances and powers to calculate final image location
    # Using a method distinct from previous implementations, considering each surface contribution
    l31 = np.zeros_like(h1)
    for i, h in enumerate(h1):
        l1 = h * d_eff1
        l2 = l1 + (d1 if d_eff1 != np.inf else 0) * (h / d_eff2)
        l3 = l2 + (d2 if d_eff2 != np.inf else 0) * (h / d_eff3)
        l31[i] = l3

    return l31


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e