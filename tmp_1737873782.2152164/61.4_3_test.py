from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def Bmat(pa):
    '''Calculate the B matrix.
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    Output
    B: a 3*3 matrix, float
    '''

    # Unpack parameters
    a, b, c, alpha, beta, gamma = pa

    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate the volume of the unit cell
    volume = a * b * c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 
                                 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
    
    # Calculate components of the B matrix
    B11 = b * c * np.sin(alpha_rad) / volume
    B22 = a * c * np.sin(beta_rad) / volume
    B33 = a * b * np.sin(gamma_rad) / volume

    B12 = (c * (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad))) / volume
    B13 = (b * (np.cos(gamma_rad) * np.cos(alpha_rad) - np.cos(beta_rad))) / volume
    B23 = (a * (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad))) / volume

    # Construct the B matrix
    B = np.array([[B11, B12, B13],
                  [0,   B22, B23],
                  [0,   0,   B33]])

    return B



def q_cal(p, b_c, det_d, p_s, wl):
    '''Calculate the momentum transfer Q at detector pixel (x,y). 
    Here we're employing the convention, k=1/\lambda,
    k represents the x-ray momentum and \lambda denotes the wavelength.
    
    Input
    p: detector pixel (x,y), a tuple of two integers
    b_c: incident beam center at detector pixel (xc,yc), a tuple of float
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''

    # Extracting pixel coordinates
    x, y = p
    xc, yc = b_c

    # Calculate the pixel displacement from the beam center
    dx = (x - xc) * p_s
    dy = (y - yc) * p_s

    # Calculate the scattered wavevector ks
    k_i = 1 / wl  # Magnitude of the incident wavevector

    # In the lab coordinate system:
    # The incident wavevector is along the +x direction, so k_i = (k_i, 0, 0)
    # Calculate the scattered wavevector components:
    ks_x = k_i * (det_d / np.sqrt(det_d**2 + dx**2 + dy**2))
    ks_y = -k_i * (dx / np.sqrt(det_d**2 + dx**2 + dy**2))
    ks_z = -k_i * (dy / np.sqrt(det_d**2 + dx**2 + dy**2))

    # Calculate the momentum transfer Q = ks - ki
    Qx = ks_x - k_i  # ki is along +x, so Qx is ks_x - ki
    Qy = ks_y  # ki has no y component, so Qy is just ks_y
    Qz = ks_z  # ki has no z component, so Qz is just ks_z

    # Return Q as a 3x1 matrix
    Q = np.array([Qx, Qy, Qz])

    return Q



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

    # Calculate the rotation angles for each frame
    theta1 = z1 * z_s
    theta2 = z2 * z_s

    # Get the B matrix from the lattice parameters
    B = Bmat(pa)

    # Calculate q vectors in Cartesian coordinates from H using the B matrix
    q1 = B @ np.array(H1)
    q2 = B @ np.array(H2)

    # Calculate the rotated q vectors
    def rotate_y(theta, vector):
        # Rotation matrix around the y-axis
        theta_rad = np.radians(theta)
        rot_matrix = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                               [0, 1, 0],
                               [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
        return rot_matrix @ vector

    q1_rotated = rotate_y(theta1, q1)
    q2_rotated = rotate_y(theta2, q2)

    # Normalize q vectors to get t_c vectors
    t1c = q1_rotated / np.linalg.norm(q1_rotated)
    t3c = np.cross(q1_rotated, q2_rotated)
    t3c /= np.linalg.norm(t3c)
    t2c = np.cross(t3c, t1c)

    # Calculate Q vectors using q_cal function
    Q1 = q_cal(p1, b_c, det_d, p_s, wl)
    Q2 = q_cal(p2, b_c, det_d, p_s, wl)

    # Normalize Q vectors to get t_g vectors
    t1g = Q1 / np.linalg.norm(Q1)
    t3g = np.cross(Q1, Q2)
    t3g /= np.linalg.norm(t3g)
    t2g = np.cross(t3g, t1g)

    # Return the orthogonal unit vectors
    t_c = (t1c, t2c, t3c)
    t_g = (t1g, t2g, t3g)

    return t_c, t_g




def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''

    # Extract the orthonormal basis vectors from the tuples
    t1c, t2c, t3c = t_c
    t1g, t2g, t3g = t_g

    # Construct matrices from the basis vectors
    Tc = np.column_stack((t1c, t2c, t3c))
    Tg = np.column_stack((t1g, t2g, t3g))

    # Calculate the orientation matrix U
    U = Tg @ np.linalg.inv(Tc)

    return U


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e