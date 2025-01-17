import numpy as np

# Background: In crystallography, the reciprocal lattice is a conceptual lattice used to understand various properties of the crystal structure. 
# The transformation from reciprocal lattice coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) involves a matrix known as the B matrix.
# The B matrix is derived from the direct lattice parameters (a, b, c, alpha, beta, gamma), where a, b, c are the lengths of the unit cell edges, 
# and alpha, beta, gamma are the angles between these edges. The reciprocal lattice vectors are defined such that they are orthogonal to the direct 
# lattice vectors, following the convention a_i · b_j = δ_ij, where δ_ij is the Kronecker delta. The transformation matrix B is constructed using 
# these parameters to convert reciprocal lattice coordinates to Cartesian coordinates.


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
    
    # Calculate the volume of the unit cell
    V = a * b * c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 
                            + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
    
    # Calculate the reciprocal lattice parameters
    a_star = b * c * np.sin(alpha_rad) / V
    b_star = a * c * np.sin(beta_rad) / V
    c_star = a * b * np.sin(gamma_rad) / V
    
    # Calculate the cosines of the angles for the reciprocal lattice
    cos_alpha_star = (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad)) / (np.sin(beta_rad) * np.sin(gamma_rad))
    cos_beta_star = (np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad)) / (np.sin(alpha_rad) * np.sin(gamma_rad))
    cos_gamma_star = (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad)) / (np.sin(alpha_rad) * np.sin(beta_rad))
    
    # Calculate the sines of the angles for the reciprocal lattice
    sin_alpha_star = np.sqrt(1 - cos_alpha_star**2)
    sin_beta_star = np.sqrt(1 - cos_beta_star**2)
    sin_gamma_star = np.sqrt(1 - cos_gamma_star**2)
    
    # Construct the B matrix
    B = np.array([
        [a_star, b_star * cos_gamma_star, c_star * cos_beta_star],
        [0, b_star * sin_gamma_star, -c_star * sin_beta_star * cos_alpha_star],
        [0, 0, c_star * sin_beta_star * sin_alpha_star]
    ])
    
    return B



# Background: In X-ray crystallography, the momentum transfer vector Q is defined as the difference between the 
# scattered wave vector (k_s) and the incident wave vector (k_i). The wave vector k is related to the wavelength 
# λ by k = 2π/λ. The detector plane can be rotated in the lab coordinate system using yaw, pitch, and roll angles, 
# which correspond to rotations around the z, y, and x axes, respectively. The transformation from detector 
# coordinates to lab coordinates involves these rotations. The incident beam is aligned with the +x direction in 
# the lab frame, and the detector plane is typically normal to this beam. The pixel position on the detector 
# (x_det, y_det) is used to calculate the scattered wave vector direction, which is then transformed into the lab 
# frame using the rotation matrices. The momentum transfer Q is then calculated in the lab frame.

def q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll):
    '''Calculate the momentum transfer Q at detector pixel (x,y). Here we use the convention of k=1/\lambda,
    k and \lambda are the x-ray momentum and wavelength respectively
    Input
    p: detector pixel (x,y), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''


    # Convert angles from degrees to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Calculate the wave number k
    k = 2 * np.pi / wl

    # Calculate the pixel position relative to the beam center
    x_det, y_det = p
    x_c, y_c = b_c
    x_rel = (x_det - x_c) * p_s
    y_rel = (y_det - y_c) * p_s

    # Calculate the scattered wave vector direction in the detector frame
    z_rel = det_d
    k_s_det = np.array([x_rel, y_rel, z_rel])
    k_s_det = k_s_det / np.linalg.norm(k_s_det) * k

    # Rotation matrices for yaw, pitch, and roll
    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    # Combined rotation matrix
    R = R_roll @ R_pitch @ R_yaw

    # Transform the scattered wave vector to the lab frame
    k_s_lab = R @ k_s_det

    # Incident wave vector in the lab frame
    k_i_lab = np.array([k, 0, 0])

    # Calculate the momentum transfer Q
    Q = k_s_lab - k_i_lab

    return Q


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('73.2', 3)
target = targets[0]

p = (1689,2527)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
yaw = 0.000730602 * 180.0 / np.pi
pitch = -0.00796329 * 180.0 / np.pi
roll = 1.51699e-5 * 180.0 / np.pi
assert np.allclose(q_cal_p(p,b_c,det_d,p_s,wl,yaw,pitch,roll), target)
target = targets[1]

p = (2190,2334)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
yaw = 0.000730602 * 180.0 / np.pi
pitch = -0.00796329 * 180.0 / np.pi
roll = 1.51699e-5 * 180.0 / np.pi
assert np.allclose(q_cal_p(p,b_c,det_d,p_s,wl,yaw,pitch,roll), target)
target = targets[2]

p = (1166,2154)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
yaw = 0.000730602 * 180.0 / np.pi
pitch = -0.00796329 * 180.0 / np.pi
roll = 1.51699e-5 * 180.0 / np.pi
assert np.allclose(q_cal_p(p,b_c,det_d,p_s,wl,yaw,pitch,roll), target)
