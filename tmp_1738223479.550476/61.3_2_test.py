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
    a, b, c, alpha, beta, gamma = pa
    
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Calculate trigonometric functions
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_alpha = np.sin(alpha_rad)
    sin_beta = np.sin(beta_rad)
    sin_gamma = np.sin(gamma_rad)
    
    # Volume of the direct lattice unit cell
    volume = a * b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    
    # Calculate the components of the B matrix using an alternative approach
    B11 = 2 * np.pi / a
    B12 = -2 * np.pi * cos_gamma / (a * sin_gamma)
    B13 = 2 * np.pi * (cos_alpha * cos_gamma - cos_beta) / (a * sin_gamma * sin_beta)
    B21 = 0
    B22 = 2 * np.pi / (b * sin_gamma)
    B23 = 2 * np.pi * (cos_beta * cos_gamma - cos_alpha) / (b * sin_gamma * sin_beta)
    B31 = 0
    B32 = 0
    B33 = 2 * np.pi / (c * sin_beta * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma))
    
    # Construct the B matrix
    B = np.array([
        [B11, B12, B13],
        [B21, B22, B23],
        [B31, B32, B33]
    ])
    
    return B


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



    # Calculate the magnitude of the wave vector
    k_magnitude = 1.0 / wl

    # Calculate the pixel offsets from the beam center in detector coordinates
    x_offset_mm = (p[0] - b_c[0]) * p_s
    y_offset_mm = (p[1] - b_c[1]) * p_s

    # Calculate the magnitude of the vector from sample to detector pixel
    distance = norm([det_d, x_offset_mm, y_offset_mm])

    # Calculate scattered wave vector components in the lab coordinate system
    k_s_x = k_magnitude * det_d / distance
    k_s_y = -k_magnitude * x_offset_mm / distance
    k_s_z = -k_magnitude * y_offset_mm / distance

    # The incident wave vector in the lab coordinate system is [k, 0, 0]
    k_i = np.array([k_magnitude, 0.0, 0.0])

    # Calculate the momentum transfer vector Q in lab coordinates
    Q = np.array([k_s_x - k_i[0], k_s_y, k_s_z])

    # Return the momentum transfer vector as a 3x1 column matrix
    return Q.reshape(3, 1)



def u_triple(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, z1, z2, z_s):


    # Function to create the B matrix for (hkl) to Cartesian transformation
    def B_matrix(pa):
        a, b, c, alpha, beta, gamma = pa
        alpha, beta, gamma = np.radians([alpha, beta, gamma])
        
        V = a * b * c * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 
                                2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        
        B11 = 2 * np.pi / a
        B12 = -2 * np.pi * np.cos(gamma) / (a * np.sin(gamma))
        B13 = a * np.cos(beta) * 2 * np.pi / V
        B22 = 2 * np.pi / (b * np.sin(gamma))
        B23 = b * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) * 2 * np.pi / (V * np.sin(gamma))
        B33 = c * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 
                          2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)) * 2 * np.pi / V
        
        return np.array([[B11, B12, B13], [0, B22, B23], [0, 0, B33]])

    # Calculate Q vector based on pixel position and detector setup
    def calculate_Q_vector(pixel, b_c, det_d, p_s, wl):
        k_magnitude = 1 / wl
        x_relative = (pixel[0] - b_c[0]) * p_s
        y_relative = (pixel[1] - b_c[1]) * p_s
        distance = np.sqrt(det_d**2 + x_relative**2 + y_relative**2)
        
        k_s_x = k_magnitude * det_d / distance
        k_s_y = -k_magnitude * x_relative / distance
        k_s_z = -k_magnitude * y_relative / distance
        
        k_i = np.array([k_magnitude, 0, 0])
        
        return np.array([k_s_x - k_i[0], k_s_y, k_s_z]).reshape(3, 1)

    # Rotation matrix around the -y axis
    def rotation_matrix_y(theta_deg):
        theta_rad = np.radians(theta_deg)
        return np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])

    # Compute the B matrix
    B = B_matrix(pa)

    # Convert reflections to Cartesian coordinates
    q1 = B @ np.array(H1).reshape(3, 1)
    q2 = B @ np.array(H2).reshape(3, 1)

    # Calculate unit vector triple in crystal coordinates
    t1c = q1 / np.linalg.norm(q1)
    t3c = np.cross(q1.flatten(), q2.flatten()).reshape(3, 1)
    t3c /= np.linalg.norm(t3c)
    t2c = np.cross(t3c.flatten(), t1c.flatten()).reshape(3, 1)

    # Momentum transfers in rotated frames
    Q1 = rotation_matrix_y(z1 * z_s) @ calculate_Q_vector(p1, b_c, det_d, p_s, wl)
    Q2 = rotation_matrix_y(z2 * z_s) @ calculate_Q_vector(p2, b_c, det_d, p_s, wl)

    # Calculate unit vector triple in lab coordinates
    t1g = Q1 / np.linalg.norm(Q1)
    t3g = np.cross(Q1.flatten(), Q2.flatten()).reshape(3, 1)
    t3g /= np.linalg.norm(t3g)
    t2g = np.cross(t3g.flatten(), t1g.flatten()).reshape(3, 1)

    # Return the orthogonal unit-vector triples
    return (t1c, t2c, t3c), (t1g, t2g, t3g)


try:
    targets = process_hdf5_to_tuple('61.3', 3)
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
    assert np.allclose(u_triple(pa,H1,H2,p1,p2,b_c,det_d,p_s,wl,z1,z2,z_s), target)

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
    assert np.allclose(u_triple(pa,H1,H2,p1,p2,b_c,det_d,p_s,wl,z1,z2,z_s), target)

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
    assert np.allclose(u_triple(pa,H1,H2,p1,p2,b_c,det_d,p_s,wl,z1,z2,z_s), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e