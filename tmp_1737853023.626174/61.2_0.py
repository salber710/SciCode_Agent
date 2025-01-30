import numpy as np

# Background: In crystallography, the reciprocal lattice is a construct used to understand the diffraction patterns of crystals. 
# The reciprocal lattice vectors are related to the direct lattice vectors through the metric tensor. 
# The transformation from reciprocal lattice coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) involves the B matrix.
# The B matrix is derived from the direct lattice parameters (a, b, c, alpha, beta, gamma), where alpha, beta, and gamma are the angles between the lattice vectors.
# The transformation is based on the relationships between the direct and reciprocal lattice vectors, where the reciprocal lattice vectors are defined as:
# b1 = 2π * (b × c) / (a · (b × c))
# b2 = 2π * (c × a) / (b · (c × a))
# b3 = 2π * (a × b) / (c · (a × b))
# The B matrix is constructed using these reciprocal lattice vectors, and it transforms (h, k, l) to Cartesian coordinates.


def Bmat(pa):
    '''Calculate the B matrix.
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    Output
    B: a 3*3 matrix, float
    '''
    a, b, c, alpha, beta, gamma = pa
    
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Validate input parameters
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Lattice parameters a, b, and c must be greater than zero.")
    if not (0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180):
        raise ValueError("Angles alpha, beta, and gamma must be between 0 and 180 degrees.")
    
    # Calculate the volume of the unit cell
    volume = a * b * c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 
                                 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
    
    # Calculate the reciprocal lattice vectors
    b1 = 2 * np.pi * np.cross([b*np.cos(gamma_rad), b*np.sin(gamma_rad), 0], [0, c*np.cos(beta_rad), c*np.sin(beta_rad)]) / volume
    b2 = 2 * np.pi * np.cross([0, c*np.cos(beta_rad), c*np.sin(beta_rad)], [a, 0, 0]) / volume
    b3 = 2 * np.pi * np.cross([a, 0, 0], [b*np.cos(gamma_rad), b*np.sin(gamma_rad), 0]) / volume
    
    # Calculate the components of the B matrix
    B = np.zeros((3, 3))
    B[:, 0] = b1
    B[:, 1] = b2
    B[:, 2] = b3
    
    # Return the B matrix
    return B



# Background: In X-ray crystallography, the momentum transfer vector Q is defined as the difference between the wave vectors of the scattered beam (k_s) and the incident beam (k_i). The wave vector k is related to the wavelength λ by k = 2π/λ. The incident beam is aligned with the +x direction in the lab coordinate system, and the detector plane is perpendicular to this direction. The detector coordinates are defined such that +x_det is aligned with -y in the lab system, and +y_det is aligned with -z in the lab system. The position of a pixel on the detector can be used to determine the direction of the scattered wave vector k_s. The momentum transfer Q is then calculated as Q = k_s - k_i, where k_i is known from the incident beam direction and wavelength.

def q_cal(p, b_c, det_d, p_s, wl):
    '''Calculate the momentum transfer Q at detector pixel (x,y). Here we're employing the convention, k=1/\lambda,
    k represents the x-ray momentum and \lambda denotes the wavelength.
    Input
    p: detector pixel (x,y), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''


    # Convert wavelength from angstroms to mm for consistency
    wl_mm = wl * 1e-10 * 1e3  # 1 angstrom = 1e-10 meters, 1 meter = 1e3 mm

    # Calculate the wave number k
    k = 2 * np.pi / wl_mm

    # Detector pixel position
    x_det, y_det = p

    # Beam center position
    x_c, y_c = b_c

    # Calculate the position of the pixel in the detector plane in mm
    x_pos = (x_det - x_c) * p_s
    y_pos = (y_det - y_c) * p_s

    # Calculate the direction of the scattered wave vector k_s
    # The detector plane is perpendicular to the incident beam, so the z component is det_d
    k_s_x = k * (x_pos / np.sqrt(x_pos**2 + y_pos**2 + det_d**2))
    k_s_y = k * (y_pos / np.sqrt(x_pos**2 + y_pos**2 + det_d**2))
    k_s_z = k * (det_d / np.sqrt(x_pos**2 + y_pos**2 + det_d**2))

    # Incident wave vector k_i is along the +x direction
    k_i = np.array([k, 0, 0])

    # Scattered wave vector k_s in the lab coordinate system
    k_s = np.array([k_s_x, k_s_y, k_s_z])

    # Calculate the momentum transfer Q
    Q = k_s - k_i

    # Return Q as a 3x1 matrix
    return Q.reshape((3, 1))

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('61.2', 3)
target = targets[0]

p = (1689,2527)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
assert np.allclose(q_cal(p,b_c,det_d,p_s,wl), target)
target = targets[1]

p = (2190,2334)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
assert np.allclose(q_cal(p,b_c,det_d,p_s,wl), target)
target = targets[2]

p = (1166,2154)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
assert np.allclose(q_cal(p,b_c,det_d,p_s,wl), target)
