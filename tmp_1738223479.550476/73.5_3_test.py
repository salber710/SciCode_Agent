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
    sin_gamma = np.sin(gamma_rad)
    
    # Calculate the volume of the unit cell
    V = a * b * c * np.sqrt(
        1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 
        2 * cos_alpha * cos_beta * cos_gamma
    )
    
    # Calculate reciprocal lattice vector magnitudes
    a_star = 2 * np.pi * b * c * np.sin(alpha_rad) / V
    b_star = 2 * np.pi * a * c * np.sin(beta_rad) / V
    c_star = 2 * np.pi * a * b * np.sin(gamma_rad) / V
    
    # Construct the B matrix using a new approach
    B = np.array([
        [a_star, 0, 0],
        [b_star * cos_gamma, b_star * sin_gamma, 0],
        [c_star * cos_beta, -c_star * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (a * b * sin_gamma)]
    ])
    
    return B


def q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll):

    
    # Helper function to create a rotation matrix from Euler angles
    def euler_to_matrix(yaw, pitch, roll):
        # Convert angles from degrees to radians
        yaw_rad, pitch_rad, roll_rad = np.deg2rad([yaw, pitch, roll])
        
        # Precompute cosines and sines
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        
        # Compute the composite rotation matrix
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr]
        ])
    
    # Get the combined rotation matrix
    R_combined = euler_to_matrix(yaw, pitch, roll)
    
    # Calculate the magnitude of the incident beam vector
    k_magnitude = 1.0 / wl
    ki = np.array([k_magnitude, 0, 0])  # Incident beam along +x axis
    
    # Calculate the detector pixel position offsets
    x_pixel, y_pixel = p
    x_center, y_center = b_c
    x_offset = (x_pixel - x_center) * p_s
    y_offset = (y_pixel - y_center) * p_s
    
    # Define the scattered beam vector in detector coordinates
    ks_detector = np.array([-y_offset, -det_d, -x_offset])
    
    # Transform scattered vector to lab coordinates
    ks_lab = R_combined @ ks_detector
    
    # Calculate the momentum transfer vector Q
    Q_vector = ks_lab - ki
    
    return Q_vector



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

    def build_reciprocal_matrix(pa):
        a, b, c, alpha, beta, gamma = pa
        alpha, beta, gamma = np.radians([alpha, beta, gamma])
        
        V = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        
        a_star = 2 * np.pi * b * c * np.sin(alpha) / V
        b_star = 2 * np.pi * a * c * np.sin(beta) / V
        c_star = 2 * np.pi * a * b * np.sin(gamma) / V
        
        return np.array([
            [a_star, 0, 0],
            [b_star * np.cos(gamma), b_star * np.sin(gamma), 0],
            [c_star * np.cos(beta), c_star * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
             V / (a * b * np.sin(gamma))]
        ])

    def calculate_momentum_transfer(p, b_c, det_d, p_s, wl, yaw, pitch, roll):
        x_pixel, y_pixel = p
        x_center, y_center = b_c
        x_offset = (x_pixel - x_center) * p_s
        y_offset = (y_pixel - y_center) * p_s
        
        yaw, pitch, roll = np.radians([yaw, pitch, roll])

        R = np.array([
            [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
            [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
        ])
        
        k_initial = np.array([1.0 / wl, 0, 0])
        k_final = np.array([-y_offset, -det_d, -x_offset])
        k_final_transformed = R @ k_final

        return k_final_transformed - k_initial

    B_matrix = build_reciprocal_matrix(pa)
    
    q1 = B_matrix @ np.array(H1)
    q2 = B_matrix @ np.array(H2)

    Q1 = calculate_momentum_transfer(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = calculate_momentum_transfer(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)

    def orthogonal_unit_vectors(v1, v2):
        v1_unit = v1 / np.linalg.norm(v1)
        v3_unit = np.cross(v1, v2)
        v3_unit /= np.linalg.norm(v3_unit)
        v2_unit = np.cross(v3_unit, v1_unit)
        return v1_unit, v2_unit, v3_unit

    t1c, t2c, t3c = orthogonal_unit_vectors(q1, q2)
    t1g, t2g, t3g = orthogonal_unit_vectors(Q1, Q2)

    t_c = (t1c.reshape(3, 1), t2c.reshape(3, 1), t3c.reshape(3, 1))
    t_g = (t1g.reshape(3, 1), t2g.reshape(3, 1), t3g.reshape(3, 1))

    return t_c, t_g


def Umat(t_c, t_g):
    '''Write down the orientation matrix which transforms from bases t_c to t_g
    Input
    t_c, tuple with three elements, each element is a 3x1 matrix, float
    t_g, tuple with three elements, each element is a 3x1 matrix, float
    Output
    U: 3x3 orthogonal matrix, float
    '''


    # Create 3x3 matrices from the input tuples
    Tc = np.column_stack(t_c)
    Tg = np.column_stack(t_g)

    # Normalize the vectors to ensure they are unit vectors
    Tc = Tc / np.linalg.norm(Tc, axis=0)
    Tg = Tg / np.linalg.norm(Tg, axis=0)

    # Use Householder reflections to enforce orthogonality
    # Householder reflection for Tc
    v_c = Tc[:,0] + np.sign(Tc[0,0]) * np.linalg.norm(Tc[:,0]) * np.eye(3)[:,0]
    Hc = np.eye(3) - 2 * np.outer(v_c, v_c) / np.dot(v_c, v_c)

    # Householder reflection for Tg
    v_g = Tg[:,0] + np.sign(Tg[0,0]) * np.linalg.norm(Tg[:,0]) * np.eye(3)[:,0]
    Hg = np.eye(3) - 2 * np.outer(v_g, v_g) / np.dot(v_g, v_g)

    # Calculate the orientation matrix U
    U = Hg @ Hc.T

    return U




def get_hkl_p(p, z, b_c, det_d, p_s, wl, yaw, pitch, roll, pa, H1, H2, p1, p2, z1, z2, z_s, chi, phi):
    '''Convert pixel (x,y) at frame z to reciprocal space (h,k,l)'''

    def euler_rotation_matrix(yaw, pitch, roll):
        yaw, pitch, roll = np.radians([yaw, pitch, roll])
        R = np.array([
            [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
            [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
        ])
        return R

    def calc_lab_q_vector(p, b_c, det_d, p_s, wl):
        x_pixel, y_pixel = p
        x_center, y_center = b_c
        x_offset = (x_pixel - x_center) * p_s
        y_offset = (y_pixel - y_center) * p_s

        k_initial = np.array([0, 0, 1.0 / wl])
        k_final = np.array([-x_offset, -y_offset, -det_d])
        return k_final - k_initial

    def generate_b_matrix(pa):
        a, b, c, alpha, beta, gamma = pa
        alpha, beta, gamma = np.radians([alpha, beta, gamma])

        V = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )

        a_star = 2 * np.pi * b * c * np.sin(alpha) / V
        b_star = 2 * np.pi * a * c * np.sin(beta) / V
        c_star = 2 * np.pi * a * b * np.sin(gamma) / V

        B = np.array([
            [a_star, 0, 0],
            [b_star * np.cos(gamma), b_star * np.sin(gamma), 0],
            [c_star * np.cos(beta), c_star * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma), V / (a * b * np.sin(gamma))]
        ])

        return B

    # Step 1: Compute the detector rotation matrix
    R_matrix = euler_rotation_matrix(yaw, pitch, roll)

    # Step 2: Calculate the lab momentum transfer vector
    Q_lab = calc_lab_q_vector(p, b_c, det_d, p_s, wl)
    Q_lab_rotated = R_matrix @ Q_lab

    # Step 3: Formulate the reciprocal lattice matrix
    B = generate_b_matrix(pa)

    # Step 4: Assume the orientation matrix U is identity for simplicity
    # In a real case, U should be calculated from known reflections
    U = np.eye(3)

    # Step 5: Transform momentum transfer vector to crystal coordinates
    Q_crystal = np.linalg.inv(U) @ Q_lab_rotated

    # Step 6: Convert to reciprocal space coordinates (h, k, l)
    hkl = np.linalg.solve(B, Q_crystal)

    return hkl


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e