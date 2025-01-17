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


# Background: In crystallography, the orientation of a crystal can be described using orthogonal unit vectors.
# For a given Bragg reflection, the momentum transfer vector Q is calculated, and the unit vector in the direction
# of Q is denoted as t1. For two non-parallel Bragg reflections, the cross product of their Q vectors gives a vector
# orthogonal to both, denoted as t3. The second unit vector t2 is orthogonal to both t1 and t3, completing the
# orthogonal set. The unit vectors t_i_c are calculated in the crystal coordinate system, where q_i are the Bragg
# reflections in Cartesian coordinates. The unit vectors t_i_g are calculated in the lab coordinate system, where
# Q_i are the momentum transfers before rotating the crystal. The transformation from reciprocal lattice coordinates
# to Cartesian coordinates is done using the B matrix, and the momentum transfer Q is calculated using the detector
# geometry and rotation angles.


def u_triple_p(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z_s, chi, phi):
    '''Calculate two orthogonal unit-vector triple t_i_c and t_i_g. Frame z starts from 0
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    H1 = (h1,k1,l1),primary reflection, h1,k1,l1 is integer
    H2 = (h2,k2,l2),secondary reflection, h2,k2,l2 is integer
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    z1,z2: frame number, integer
    z_s: step size in the \phi rotation, float in the unit of degree
    chi,phi: diffractometer angles, float in the unit of degree
    Output
    t_c_t_g: tuple (t_c,t_g), t_c = (t1c,t2c,t3c) and t_g = (t1g,t2g,t3g).
    Each element inside t_c and t_g is a 3x1 matrix, float
    '''

    # Calculate the B matrix for the transformation from reciprocal to Cartesian coordinates
    B = Bmat(pa)

    # Convert the reciprocal lattice vectors to Cartesian coordinates
    q1 = B @ np.array(H1)
    q2 = B @ np.array(H2)

    # Calculate the unit vectors in the crystal coordinate system
    t1c = q1 / np.linalg.norm(q1)
    t3c = np.cross(q1, q2)
    t3c /= np.linalg.norm(t3c)
    t2c = np.cross(t3c, t1c)

    # Calculate the momentum transfer Q for the primary and secondary reflections
    Q1 = q_cal_p(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = q_cal_p(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)

    # Calculate the unit vectors in the lab coordinate system
    t1g = Q1 / np.linalg.norm(Q1)
    t3g = np.cross(Q1, Q2)
    t3g /= np.linalg.norm(t3g)
    t2g = np.cross(t3g, t1g)

    # Return the orthogonal unit-vector triples
    t_c = (t1c, t2c, t3c)
    t_g = (t1g, t2g, t3g)

    return t_c, t_g


# Background: In crystallography, the orientation matrix U is used to transform between two sets of orthogonal 
# unit vectors. Specifically, it transforms from the crystal coordinate system, represented by the unit vectors 
# {t_i^c}, to the lab coordinate system, represented by the unit vectors {t_i^g}. The matrix U is a 3x3 orthogonal 
# matrix that aligns the basis vectors of one system with those of the other. This transformation is crucial for 
# understanding the orientation of the crystal in the lab frame, which is necessary for interpreting diffraction 
# patterns and other crystallographic data. The matrix U is constructed by aligning the corresponding unit vectors 
# from the two systems.

def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''
    # Construct the matrix from the crystal coordinate system
    Tc = np.column_stack(t_c)

    # Construct the matrix from the lab coordinate system
    Tg = np.column_stack(t_g)

    # Calculate the orientation matrix U
    U = Tg @ np.linalg.inv(Tc)

    return U



# Background: In crystallography, converting detector pixel coordinates to reciprocal space coordinates (h, k, l) 
# involves using the orientation matrix U and the transformation matrix B. The pixel coordinates (x_det, y_det) 
# are first used to calculate the momentum transfer vector Q in the lab frame. This vector is then transformed 
# into the crystal frame using the inverse of the orientation matrix U. The resulting vector is in Cartesian 
# coordinates and is further transformed into reciprocal lattice coordinates using the inverse of the B matrix. 
# This process allows us to determine the (h, k, l) indices corresponding to a given pixel on the detector, 
# which is crucial for analyzing diffraction patterns and understanding the crystal structure.

def get_hkl_p(p, z, b_c, det_d, p_s, wl, yaw, pitch, roll, pa, H1, H2, p1, p2, z1, z2, z_s, chi, phi):
    '''Convert pixel (x,y) at frame z to reciprocal space (h,k,l)
    Input
    p: detector pixel (x,y), a tuple of two integer
    z: frame number, integer
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    yaw,pitch,roll: rotation angles of the detector, float in the unit of degree
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    H1 = (h1,k1,l1),primary reflection, h1,k1,l1 is integer
    H2 = (h2,k2,l2),secondary reflection, h2,k2,l2 is integer
    p1: detector pixel (x1,y1), a tuple of two integer
    p2: detector pixel (x2,y2), a tuple of two integer
    z1,z2: frame number, integer
    z_s: step size in the \phi rotation, float in the unit of degree
    chi,phi: diffractometer angles, float in the unit of degree
    Output
    q: 3x1 orthogonal matrix, float
    '''


    # Calculate the B matrix for the transformation from reciprocal to Cartesian coordinates
    B = Bmat(pa)

    # Calculate the orthogonal unit-vector triples t_c and t_g
    t_c, t_g = u_triple_p(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z_s, chi, phi)

    # Calculate the orientation matrix U
    U = Umat(t_c, t_g)

    # Calculate the momentum transfer Q for the given pixel and frame
    Q = q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll)

    # Transform Q from the lab frame to the crystal frame using the inverse of U
    Q_crystal = np.linalg.inv(U) @ Q

    # Transform Q_crystal from Cartesian to reciprocal lattice coordinates using the inverse of B
    hkl = np.linalg.inv(B) @ Q_crystal

    return hkl


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('73.5', 3)
target = targets[0]

a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
pa = (a,b,c,alpha,beta,gamma)
H1 = (1,1,1)
H2 = (2,2,0)
p1 = (1689,2527)
p2 = (2190,2334)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
yaw = 0.000730602 * 180.0 / np.pi
pitch = -0.00796329 * 180.0 / np.pi
roll = 1.51699e-5 * 180.0 / np.pi
z1 = 132-1
z2 = 225-1
z_s = 0.05
chi = 0
phi = 0
p = (1166,2154)
z = 329-1
assert np.allclose(get_hkl_p(p,z,b_c,det_d,p_s,wl,yaw,pitch,roll,pa,H1,H2,p1,p2,z1,z2,z_s,chi,phi), target)
target = targets[1]

a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
pa = (a,b,c,alpha,beta,gamma)
H1 = (1,1,1)
H2 = (2,2,0)
p1 = (1689,2527)
p2 = (2190,2334)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
yaw = 0.000730602 * 180.0 / np.pi
pitch = -0.00796329 * 180.0 / np.pi
roll = 1.51699e-5 * 180.0 / np.pi
z1 = 132-1
z2 = 225-1
z_s = 0.05
chi = 0
phi = 0
p = (632,1060)
z = 232-1
assert np.allclose(get_hkl_p(p,z,b_c,det_d,p_s,wl,yaw,pitch,roll,pa,H1,H2,p1,p2,z1,z2,z_s,chi,phi), target)
target = targets[2]

a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
pa = (a,b,c,alpha,beta,gamma)
H1 = (1,1,1)
H2 = (2,2,0)
p1 = (1689,2527)
p2 = (2190,2334)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
yaw = 0.000730602 * 180.0 / np.pi
pitch = -0.00796329 * 180.0 / np.pi
roll = 1.51699e-5 * 180.0 / np.pi
z1 = 132-1
z2 = 225-1
z_s = 0.05
chi = 0
phi = 0
p = (1999,343)
z = 259-1
assert np.allclose(get_hkl_p(p,z,b_c,det_d,p_s,wl,yaw,pitch,roll,pa,H1,H2,p1,p2,z1,z2,z_s,chi,phi), target)
