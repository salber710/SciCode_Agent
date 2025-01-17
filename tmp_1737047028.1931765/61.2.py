import numpy as np

# Background: In crystallography, the reciprocal lattice is a construct used to understand the diffraction patterns of crystals. 
# The reciprocal lattice vectors are defined in terms of the direct lattice vectors. The transformation from reciprocal lattice 
# coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) involves a matrix, B, which is derived from the direct lattice 
# parameters (a, b, c, alpha, beta, gamma). The direct lattice parameters describe the geometry of the unit cell in real space, 
# where a, b, and c are the lengths of the cell edges, and alpha, beta, and gamma are the angles between these edges. 
# The reciprocal lattice vectors are defined such that a_i · b_j = δ_ij, where δ_ij is the Kronecker delta. 
# The transformation matrix B is constructed using these parameters and is used to convert reciprocal lattice coordinates 
# to Cartesian coordinates. The angles alpha, beta, and gamma are converted from degrees to radians for trigonometric calculations.


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
    
    # Calculate the volume of the unit cell in reciprocal space
    V = a * b * c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 
                            + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
    
    # Calculate the reciprocal lattice parameters
    a_star = b * c * np.sin(alpha_rad) / V
    b_star = a * c * np.sin(beta_rad) / V
    c_star = a * b * np.sin(gamma_rad) / V
    
    # Calculate the components of the B matrix
    B = np.array([
        [a_star, b_star * np.cos(gamma_rad), c_star * np.cos(beta_rad)],
        [0, b_star * np.sin(gamma_rad), -c_star * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)],
        [0, 0, c_star * V / (a * b * np.sin(gamma_rad))]
    ])
    
    return B



# Background: In X-ray crystallography, the momentum transfer vector Q is defined as the difference between the wave vectors 
# of the scattered beam (k_s) and the incident beam (k_i). The wave vector k is related to the wavelength λ by k = 1/λ. 
# The incident beam is aligned with the +x direction in the lab coordinate system, and the detector plane is perpendicular 
# to this beam. The detector coordinates are defined such that +x_det is aligned with -y in the lab system, and +y_det is 
# aligned with -z in the lab system. To calculate Q, we need to determine the scattered wave vector k_s based on the 
# position of a pixel on the detector, the center of the beam on the detector, the distance from the sample to the detector, 
# and the pixel size. The incident wave vector k_i is straightforward as it is aligned with the incident beam direction.

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


    # Convert wavelength from angstrom to mm for consistency with other units
    wl_mm = wl * 1e-4  # 1 angstrom = 0.1 nm = 0.0001 mm

    # Calculate the wave vector magnitude
    k_magnitude = 1 / wl_mm

    # Detector pixel coordinates
    x_det, y_det = p
    xc, yc = b_c

    # Calculate the position of the pixel in the detector plane in mm
    x_pos = (x_det - xc) * p_s
    y_pos = (y_det - yc) * p_s

    # Calculate the scattered wave vector k_s in the lab coordinate system
    # k_s = (k_x, k_y, k_z) in lab coordinates
    k_x = k_magnitude * det_d / np.sqrt(det_d**2 + x_pos**2 + y_pos**2)
    k_y = -k_magnitude * x_pos / np.sqrt(det_d**2 + x_pos**2 + y_pos**2)
    k_z = -k_magnitude * y_pos / np.sqrt(det_d**2 + x_pos**2 + y_pos**2)

    # Incident wave vector k_i in the lab coordinate system
    k_i = np.array([k_magnitude, 0, 0])

    # Scattered wave vector k_s in the lab coordinate system
    k_s = np.array([k_x, k_y, k_z])

    # Calculate the momentum transfer Q = k_s - k_i
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
