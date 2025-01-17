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


# Background: In crystallography, the orientation of a crystal can be described using orthogonal unit vectors. 
# When considering Bragg reflections, the vectors q_i represent the reflections in Cartesian coordinates. 
# The unit vector triple {t_i^c} is defined such that t_1^c is parallel to q_1, t_3^c is parallel to the cross product 
# of q_1 and q_2, and t_2^c is orthogonal to both, completing the right-handed system. Similarly, {t_i^g} is defined 
# using the momentum transfer vectors Q_i, where t_1^g is parallel to Q_1, t_3^g is parallel to the cross product 
# of Q_1 and Q_2, and t_2^g is orthogonal to both. These unit vectors are crucial for understanding the orientation 
# of the crystal and the geometry of the diffraction pattern.


def u_triple(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, z1, z2, z_s):
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
    z1,z2: frame number, integer
    z_s: step size in the \phi rotation, float in the unit of degree
    Output
    t_c_t_g: tuple (t_c,t_g), t_c = (t1c,t2c,t3c) and t_g = (t1g,t2g,t3g).
    Each element inside t_c and t_g is a 3x1 matrix, float
    '''

    # Calculate the B matrix for the transformation
    B = Bmat(pa)

    # Calculate q vectors in Cartesian coordinates
    q1 = np.dot(B, H1)
    q2 = np.dot(B, H2)

    # Normalize q1 and q2 to get t1c and t2c
    t1c = q1 / np.linalg.norm(q1)
    t2c = q2 / np.linalg.norm(q2)

    # Calculate t3c as the cross product of t1c and t2c
    t3c = np.cross(t1c, t2c)
    t3c /= np.linalg.norm(t3c)

    # Recalculate t2c to ensure orthogonality
    t2c = np.cross(t3c, t1c)
    t2c /= np.linalg.norm(t2c)

    # Calculate Q vectors for the given frames
    Q1 = q_cal(p1, b_c, det_d, p_s, wl)
    Q2 = q_cal(p2, b_c, det_d, p_s, wl)

    # Normalize Q1 and Q2 to get t1g and t2g
    t1g = Q1 / np.linalg.norm(Q1)
    t2g = Q2 / np.linalg.norm(Q2)

    # Calculate t3g as the cross product of t1g and t2g
    t3g = np.cross(t1g.flatten(), t2g.flatten())
    t3g /= np.linalg.norm(t3g)

    # Recalculate t2g to ensure orthogonality
    t2g = np.cross(t3g, t1g.flatten())
    t2g /= np.linalg.norm(t2g)

    # Reshape to 3x1 matrices
    t1c = t1c.reshape((3, 1))
    t2c = t2c.reshape((3, 1))
    t3c = t3c.reshape((3, 1))
    t1g = t1g.reshape((3, 1))
    t2g = t2g.reshape((3, 1))
    t3g = t3g.reshape((3, 1))

    # Return the unit vector triples
    t_c = (t1c, t2c, t3c)
    t_g = (t1g, t2g, t3g)

    return (t_c, t_g)



# Background: In crystallography, the orientation matrix U is a unitary transformation matrix that describes the rotation 
# needed to align one set of orthogonal basis vectors with another. Specifically, it transforms the basis vectors 
# {t_i^c}, which are aligned with the crystal's Cartesian coordinates, to the basis vectors {t_i^g}, which are aligned 
# with the laboratory's Cartesian coordinates. The matrix U is orthogonal, meaning its columns (or rows) are orthonormal 
# vectors, and it satisfies the condition U^T * U = I, where I is the identity matrix. The matrix U can be constructed 
# by arranging the vectors of the target basis {t_i^g} as columns and expressing them in terms of the initial basis 
# {t_i^c}.

def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''


    # Extract the vectors from the tuples
    t1c, t2c, t3c = t_c
    t1g, t2g, t3g = t_g

    # Construct the matrix T_c and T_g by stacking the vectors as columns
    T_c = np.hstack((t1c, t2c, t3c))
    T_g = np.hstack((t1g, t2g, t3g))

    # Calculate the orientation matrix U as the product of T_g and the inverse of T_c
    U = np.dot(T_g, np.linalg.inv(T_c))

    return U


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('61.4', 3)
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
z1 = 132-1
z2 = 225-1
z_s = 0.05
t_c,t_g = u_triple(pa,H1,H2,p1,p2,b_c,det_d,p_s,wl,z1,z2,z_s)
assert np.allclose(Umat(t_c,t_g), target)
target = targets[1]

a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
pa = (a,b,c,alpha,beta,gamma)
H1 = (1,1,3)
H2 = (2,2,0)
p1 = (1166,2154)
p2 = (2190,2334)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
z1 = 329-1
z2 = 225-1
z_s = 0.05
t_c,t_g = u_triple(pa,H1,H2,p1,p2,b_c,det_d,p_s,wl,z1,z2,z_s)
assert np.allclose(Umat(t_c,t_g), target)
target = targets[2]

a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
pa = (a,b,c,alpha,beta,gamma)
H1 = (1,1,1)
H2 = (3,1,5)
p1 = (1689,2527)
p2 = (632,1060)
b_c = (1699.85, 3037.62)
det_d = 219.741
p_s = 0.1
wl = 0.710511
z1 = 132-1
z2 = 232-1
z_s = 0.05
t_c,t_g = u_triple(pa,H1,H2,p1,p2,b_c,det_d,p_s,wl,z1,z2,z_s)
assert np.allclose(Umat(t_c,t_g), target)
