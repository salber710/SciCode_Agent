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



# Background: In crystallography, understanding the orientation of crystal planes in relation to the lab coordinate system is crucial for interpreting diffraction patterns. The orthogonal unit-vector triples are used for this purpose.
# 
# The unit-vector triple {$\mathbf{\hat{t}}_i^c$} is defined in the crystal coordinate system:
# - $\mathbf{\hat{t}}_1^c$ is aligned with the direction of the reciprocal lattice vector $q_1$.
# - $\mathbf{\hat{t}}_3^c$ is aligned with the normal to the plane formed by $q_1$ and $q_2$, i.e., $q_1 \times q_2$.
# - $\mathbf{\hat{t}}_2^c$ is perpendicular to both $\mathbf{\hat{t}}_1^c$ and $\mathbf{\hat{t}}_3^c$, completing the right-handed system.
#
# Similarly, {$\mathbf{\hat{t}}_i^g$} is defined in the lab coordinate system:
# - $\mathbf{\hat{t}}_1^g$ is aligned with the momentum transfer vector $Q_1$.
# - $\mathbf{\hat{t}}_3^g$ is aligned with the normal to the plane formed by $Q_1$ and $Q_2$, i.e., $Q_1 \times Q_2$.
# - $\mathbf{\hat{t}}_2^g$ is perpendicular to both $\mathbf{\hat{t}}_1^g$ and $\mathbf{\hat{t}}_3^g$.
#
# The function calculates these vectors using the given parameters, including the transformation from reciprocal lattice coordinates to Cartesian coordinates and the detector setup.

def u_triple(pa, H1, H2, p1, p2, b_c, det_d, p_s, wl, z1, z2, z_s):
    '''Calculate two orthogonal unit-vector triples t_i_c and t_i_g. Frame z starts from 0
    Input
    pa = (a,b,c,alpha,beta,gamma)
    H1 = (h1,k1,l1), primary reflection
    H2 = (h2,k2,l2), secondary reflection
    p1: detector pixel (x1,y1)
    p2: detector pixel (x2,y2)
    b_c: incident beam center at detector pixel (xc,yc)
    det_d: sample distance to the detector
    p_s: detector pixel size
    wl: X-ray wavelength
    z1,z2: frame number
    z_s: step size in the \phi rotation
    Output
    t_c_t_g: tuple (t_c,t_g), t_c = (t1c,t2c,t3c) and t_g = (t1g,t2g,t3g).
    Each element inside t_c and t_g is a 3x1 matrix, float
    '''


    # Calculate the B matrix for transforming (h,k,l) to Cartesian (q_x, q_y, q_z)
    def Bmat(pa):
        a, b, c, alpha, beta, gamma = pa
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
        sin_alpha = np.sin(alpha_rad)
        sin_beta = np.sin(beta_rad)
        sin_gamma = np.sin(gamma_rad)
        B11 = 2 * np.pi / a
        B12 = -2 * np.pi * cos_gamma / (a * sin_gamma)
        B13 = 2 * np.pi * (cos_alpha * cos_gamma - cos_beta) / (a * sin_gamma * sin_beta)
        B22 = 2 * np.pi / (b * sin_gamma)
        B23 = 2 * np.pi * (cos_beta * cos_gamma - cos_alpha) / (b * sin_gamma * sin_beta)
        B33 = 2 * np.pi / (c * sin_beta * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma))
        return np.array([
            [B11, B12, B13],
            [0, B22, B23],
            [0, 0, B33]
        ])
    
    # Calculate the momentum transfer Q using the provided function
    def q_cal(p, b_c, det_d, p_s, wl):
        k_magnitude = 1.0 / wl
        x_offset_mm = (p[0] - b_c[0]) * p_s
        y_offset_mm = (p[1] - b_c[1]) * p_s
        distance = np.linalg.norm([det_d, x_offset_mm, y_offset_mm])
        k_s_x = k_magnitude * det_d / distance
        k_s_y = -k_magnitude * x_offset_mm / distance
        k_s_z = -k_magnitude * y_offset_mm / distance
        k_i = np.array([k_magnitude, 0.0, 0.0])
        Q = np.array([k_s_x - k_i[0], k_s_y, k_s_z])
        return Q.reshape(3, 1)

    # Calculate B matrix
    B = Bmat(pa)

    # Calculate q1 and q2 in Cartesian coordinates
    q1 = B @ np.array(H1).reshape(3, 1)
    q2 = B @ np.array(H2).reshape(3, 1)

    # Calculate t_i_c
    t1c = q1 / np.linalg.norm(q1)
    t3c = np.cross(q1.flatten(), q2.flatten()).reshape(3, 1)
    t3c = t3c / np.linalg.norm(t3c)
    t2c = np.cross(t3c.flatten(), t1c.flatten()).reshape(3, 1)

    # Calculate Q1 and Q2 for the respective frames
    theta1 = z1 * z_s
    theta2 = z2 * z_s

    # Assuming rotation matrix about -y axis for each frame
    def rotation_matrix_y(theta):
        theta_rad = np.radians(theta)
        return np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])

    R1 = rotation_matrix_y(theta1)
    R2 = rotation_matrix_y(theta2)

    Q1 = R1 @ q_cal(p1, b_c, det_d, p_s, wl)
    Q2 = R2 @ q_cal(p2, b_c, det_d, p_s, wl)

    # Calculate t_i_g
    t1g = Q1 / np.linalg.norm(Q1)
    t3g = np.cross(Q1.flatten(), Q2.flatten()).reshape(3, 1)
    t3g = t3g / np.linalg.norm(t3g)
    t2g = np.cross(t3g.flatten(), t1g.flatten()).reshape(3, 1)

    # Return the orthogonal unit-vector triples
    t_c = (t1c, t2c, t3c)
    t_g = (t1g, t2g, t3g)

    return (t_c, t_g)


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