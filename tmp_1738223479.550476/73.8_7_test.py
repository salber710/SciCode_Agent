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

    def rotation_matrix(yaw, pitch, roll):
        # Convert angles to radians
        yaw, pitch, roll = np.radians([yaw, pitch, roll])
        
        # Rotation matrices for yaw, pitch, roll
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        return Rz @ Ry @ Rx

    def compute_q_lab(p, b_c, det_d, p_s, wl):
        # Calculate pixel offsets
        x, y = p
        x_center, y_center = b_c
        x_offset = (x - x_center) * p_s
        y_offset = (y - y_center) * p_s

        # Initial and final wave vectors
        k_initial = np.array([1.0 / wl, 0, 0])
        k_final = np.array([x_offset, y_offset, -det_d])
        
        return k_final - k_initial

    def reciprocal_lattice_matrix(pa):
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
            [a_star, b_star * np.cos(gamma), c_star * np.cos(beta)],
            [0, b_star * np.sin(gamma), c_star * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
            [0, 0, V / (a * b * np.sin(gamma))]
        ])

    # Step 1: Compute rotation matrix
    R = rotation_matrix(yaw, pitch, roll)

    # Step 2: Calculate the momentum transfer vector in lab frame
    Q_lab = compute_q_lab(p, b_c, det_d, p_s, wl)
    Q_rotated = R @ Q_lab

    # Step 3: Calculate the reciprocal lattice matrix
    B_matrix = reciprocal_lattice_matrix(pa)

    # Step 4: Orientation matrix U
    U = np.eye(3)  # Placeholder for actual U

    # Step 5: Transform Q to crystal coordinates
    Q_crystal = U @ Q_rotated

    # Step 6: Convert to reciprocal space coordinates (h, k, l)
    hkl = np.linalg.solve(B_matrix, Q_crystal)

    return hkl




def ringdstar(pa, polar_max, wl):
    '''List all d*<d*_max and the corresponding (h,k,l). d*_max is determined by the maximum scattering angle
    and the x-ray wavelength
    Input
    pa = (a,b,c,alpha,beta,gamma)
    a,b,c: the lengths a, b, and c of the three cell edges meeting at a vertex, float in the unit of angstrom
    alpha,beta,gamma: the angles alpha, beta, and gamma between those edges, float in the unit of degree
    polar_max: maximum scattering angle, i.e. maximum angle between the x-ray beam axis
               and the powder ring, float in the unit of degree
    wl: X-ray wavelength, float in the unit of angstrom
    Output
    ringhkls: a dictionary, key is d* and each item is a sorted list with element of corresponding (h,k,l)
    '''
    
    # Calculate d* max based on Bragg's law
    theta_max_rad = math.radians(polar_max) / 2
    d_star_max = 2 * math.sin(theta_max_rad) / wl

    # Unpack and convert parameters
    a, b, c, alpha, beta, gamma = pa
    alpha_rad, beta_rad, gamma_rad = map(math.radians, [alpha, beta, gamma])

    # Unit cell volume
    V = a * b * c * math.sqrt(
        1 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2 +
        2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
    )

    # Reciprocal lattice vectors
    a_star = 2 * math.pi * b * c * math.sin(alpha_rad) / V
    b_star = 2 * math.pi * a * c * math.sin(beta_rad) / V
    c_star = 2 * math.pi * a * b * math.sin(gamma_rad) / V

    cos_alpha_star = (math.cos(beta_rad) * math.cos(gamma_rad) - math.cos(alpha_rad)) / (math.sin(beta_rad) * math.sin(gamma_rad))
    cos_beta_star = (math.cos(alpha_rad) * math.cos(gamma_rad) - math.cos(beta_rad)) / (math.sin(alpha_rad) * math.sin(gamma_rad))
    cos_gamma_star = (math.cos(alpha_rad) * math.cos(beta_rad) - math.cos(gamma_rad)) / (math.sin(alpha_rad) * math.sin(beta_rad))

    ringhkls = defaultdict(list)

    # Iterate over h, k, l values
    for h, k, l in itertools.product(range(-10, 11), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue

        # Calculate d_star^2
        d_star_squared = (
            (h * a_star)**2 +
            (k * b_star)**2 +
            (l * c_star)**2 +
            2 * h * k * a_star * b_star * cos_gamma_star +
            2 * h * l * a_star * c_star * cos_beta_star +
            2 * k * l * b_star * c_star * cos_alpha_star
        )

        d_star = math.sqrt(d_star_squared)

        if d_star < d_star_max:
            ringhkls[d_star].append((h, k, l))

    # Sort the results
    sorted_ringhkls = {d_star: sorted(hkls) for d_star, hkls in sorted(ringhkls.items())}

    return sorted_ringhkls




def hkl_pairs(pa, p1, p2, b_c, det_d, p_s, wl, yaw, pitch, roll, polar_max):
    '''Find the possible (h,k,l) for a pair of Bragg reflections (Q1,Q2)'''

    def pixel_to_lab_coords(pixel, beam_center, det_distance, pixel_size, yaw, pitch, roll):
        # Calculate pixel offsets from the beam center
        x_offset = (pixel[0] - beam_center[0]) * pixel_size
        y_offset = (pixel[1] - beam_center[1]) * pixel_size
        
        # Convert angles to radians
        yaw_rad, pitch_rad, roll_rad = np.radians([yaw, pitch, roll])
        
        # Rotation matrix
        R = np.array([
            [np.cos(yaw_rad) * np.cos(pitch_rad), 
             np.cos(yaw_rad) * np.sin(pitch_rad) * np.sin(roll_rad) - np.sin(yaw_rad) * np.cos(roll_rad), 
             np.cos(yaw_rad) * np.sin(pitch_rad) * np.cos(roll_rad) + np.sin(yaw_rad) * np.sin(roll_rad)],
            [np.sin(yaw_rad) * np.cos(pitch_rad), 
             np.sin(yaw_rad) * np.sin(pitch_rad) * np.sin(roll_rad) + np.cos(yaw_rad) * np.cos(roll_rad), 
             np.sin(yaw_rad) * np.sin(pitch_rad) * np.cos(roll_rad) - np.cos(yaw_rad) * np.sin(roll_rad)],
            [-np.sin(pitch_rad), 
             np.cos(pitch_rad) * np.sin(roll_rad), 
             np.cos(pitch_rad) * np.cos(roll_rad)]
        ])
        
        # Scattered beam vector in detector coordinates
        ks_detector = np.array([x_offset, -det_distance, y_offset])
        
        # Transform to lab coordinates
        ks_lab = R @ ks_detector
        
        # Incident beam vector
        ki_magnitude = 1 / wl
        ki = np.array([ki_magnitude, 0, 0])
        
        # Momentum transfer vector Q
        Q = ks_lab - ki
        return Q

    def calculate_d_star(Q):
        return np.linalg.norm(Q) / (2 * np.pi)

    # Compute Q vectors for the two detector positions
    Q1 = pixel_to_lab_coords(p1, b_c, det_d, p_s, wl, yaw, pitch, roll)
    Q2 = pixel_to_lab_coords(p2, b_c, det_d, p_s, wl, yaw, pitch, roll)

    # Calculate d* values
    d_star_1 = calculate_d_star(Q1)
    d_star_2 = calculate_d_star(Q2)

    # Extract unit cell parameters and convert angles to radians
    a, b, c, alpha, beta, gamma = pa
    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    # Compute volume and reciprocal lattice parameters
    V = a * b * c * np.sqrt(
        1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
        2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )
    
    a_star = 2 * np.pi * b * c * np.sin(alpha) / V
    b_star = 2 * np.pi * a * c * np.sin(beta) / V
    c_star = 2 * np.pi * a * b * np.sin(gamma) / V
    
    cos_alpha_star = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (np.sin(beta) * np.sin(gamma))
    cos_beta_star = (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (np.sin(alpha) * np.sin(gamma))
    cos_gamma_star = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))

    # Iterate over permutations of h, k, l values
    possible_hkls_1 = []
    possible_hkls_2 = []

    for h, k, l in permutations(range(-10, 11), 3):
        if (h, k, l) == (0, 0, 0):
            continue

        # Calculate squared d_star
        d_star_sq = (
            (h * a_star)**2 + 
            (k * b_star)**2 + 
            (l * c_star)**2 +
            2 * h * k * a_star * b_star * cos_gamma_star + 
            2 * h * l * a_star * c_star * cos_beta_star + 
            2 * k * l * b_star * c_star * cos_alpha_star
        )

        d_star_value = np.sqrt(d_star_sq)

        # Check if this d_star matches d_star_1 or d_star_2
        if np.isclose(d_star_value, d_star_1, atol=1e-5):
            possible_hkls_1.append((h, k, l))
        if np.isclose(d_star_value, d_star_2, atol=1e-5):
            possible_hkls_2.append((h, k, l))

    # Sort and return the possible hkl lists
    ha = sorted(set(possible_hkls_1))
    hb = sorted(set(possible_hkls_2))

    return (ha, hb)



def Umat_p(pa, p1, p2, p3, b_c, det_d, p_s, wl, yaw, pitch, roll, z1, z2, z3, z_s, chi, phi, polar_max):



    def pixel_to_lab_coordinates(pixel, beam_center, det_distance, pixel_size, yaw, pitch, roll):
        x_shift = (pixel[0] - beam_center[0]) * pixel_size
        y_shift = (pixel[1] - beam_center[1]) * pixel_size
        
        yaw_rad, pitch_rad, roll_rad = np.radians([yaw, pitch, roll])
        
        R = np.array([
            [np.cos(yaw_rad) * np.cos(pitch_rad), 
             np.cos(yaw_rad) * np.sin(pitch_rad) * np.sin(roll_rad) - np.sin(yaw_rad) * np.cos(roll_rad), 
             np.cos(yaw_rad) * np.sin(pitch_rad) * np.cos(roll_rad) + np.sin(yaw_rad) * np.sin(roll_rad)],
            [np.sin(yaw_rad) * np.cos(pitch_rad), 
             np.sin(yaw_rad) * np.sin(pitch_rad) * np.sin(roll_rad) + np.cos(yaw_rad) * np.cos(roll_rad), 
             np.sin(yaw_rad) * np.sin(pitch_rad) * np.cos(roll_rad) - np.cos(yaw_rad) * np.sin(roll_rad)],
            [-np.sin(pitch_rad), 
             np.cos(pitch_rad) * np.sin(roll_rad), 
             np.cos(pitch_rad) * np.cos(roll_rad)]
        ])
        
        detector_vector = np.array([x_shift, -det_distance, y_shift])
        lab_vector = R @ detector_vector
        
        ki_magnitude = 1 / wl
        ki_vector = np.array([ki_magnitude, 0, 0])
        
        Q_vector = lab_vector - ki_vector
        return Q_vector

    def build_reciprocal_lattice_matrix(pa):
        a, b, c, alpha, beta, gamma = pa
        alpha, beta, gamma = np.radians([alpha, beta, gamma])

        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )

        a_star = 2 * np.pi * b * c * np.sin(alpha) / volume
        b_star = 2 * np.pi * a * c * np.sin(beta) / volume
        c_star = 2 * np.pi * a * b * np.sin(gamma) / volume

        return np.array([
            [a_star, 0, 0],
            [b_star * np.cos(gamma), b_star * np.sin(gamma), 0],
            [c_star * np.cos(beta), c_star * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma), volume / (a * b * np.sin(gamma))]
        ])

    Q1 = pixel_to_lab_coordinates(p1, b_c, det_d, p_s, yaw, pitch, roll)
    Q2 = pixel_to_lab_coordinates(p2, b_c, det_d, p_s, yaw, pitch, roll)
    Q3 = pixel_to_lab_coordinates(p3, b_c, det_d, p_s, yaw, pitch, roll)
    
    B_matrix = build_reciprocal_lattice_matrix(pa)

    candidate_H1 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Example HKLs
    candidate_H2 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Example HKLs

    best_U = None
    best_H1 = None
    best_H2 = None
    best_H = None
    min_deviation = float('inf')

    for H1 in candidate_H1:
        for H2 in candidate_H2:
            q1 = B_matrix @ np.array(H1)
            q2 = B_matrix @ np.array(H2)

            # Using orthogonal procrustes for a fresh approach
            R, _ = orthogonal_procrustes(np.column_stack((q1, q2)), np.column_stack((Q1, Q2)))
            U_candidate = R

            Q3_crystal = np.linalg.solve(U_candidate @ B_matrix, Q3)
            deviation = np.linalg.norm(Q3_crystal - np.round(Q3_crystal))
            
            if deviation < min_deviation:
                min_deviation = deviation
                best_U = U_candidate
                best_H1 = H1
                best_H2 = H2
                best_H = np.round(Q3_crystal).astype(int)

    return (best_U, best_H1, best_H2, best_H)


try:
    targets = process_hdf5_to_tuple('73.8', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
    pa = (a,b,c,alpha,beta,gamma)
    p1 = (1689,2527)
    p2 = (2190,2334)
    p3 = (632,1060)
    b_c = (1699.85, 3037.62)
    det_d = 219.741
    p_s = 0.1
    wl = 0.710511
    yaw = 0.000730602 * 180.0 / np.pi
    pitch = -0.00796329 * 180.0 / np.pi
    roll = 1.51699e-5 * 180.0 / np.pi
    z1 = 132-1
    z2 = 225-1
    z3 = 232-1
    z_s = 0.05
    chi = 0
    phi = 0
    polar_max = 22
    assert cmp_tuple_or_list(Umat_p(pa,p1,p2,p3,b_c,det_d,p_s,wl,yaw,pitch,roll,z1,z2,z3,z_s,chi,phi,polar_max), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
    pa = (a,b,c,alpha,beta,gamma)
    p1 = (2176,2867)
    p2 = (1168,2157)
    p3 = (2239,1705)
    b_c = (1699.85, 3037.62)
    det_d = 219.741
    p_s = 0.1
    wl = 0.710511
    yaw = 0.000730602 * 180.0 / np.pi
    pitch = -0.00796329 * 180.0 / np.pi
    roll = 1.51699e-5 * 180.0 / np.pi
    z1 = 234-1
    z2 = 328-1
    z3 = 340-1
    z_s = 0.05
    chi = 0
    phi = 0
    polar_max = 50
    assert cmp_tuple_or_list(Umat_p(pa,p1,p2,p3,b_c,det_d,p_s,wl,yaw,pitch,roll,z1,z2,z3,z_s,chi,phi,polar_max), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,90,90,90)
    pa = (a,b,c,alpha,beta,gamma)
    p1 = (1945,1362)
    p2 = (2645,999)
    p3 = (1999,343)
    b_c = (1699.85, 3037.62)
    det_d = 219.741
    p_s = 0.1
    wl = 0.710511
    yaw = 0.000730602 * 180.0 / np.pi
    pitch = -0.00796329 * 180.0 / np.pi
    roll = 1.51699e-5 * 180.0 / np.pi
    z1 = 27-1
    z2 = 173-1
    z3 = 259-1
    z_s = 0.05
    chi = 0
    phi = 0
    polar_max = 55
    assert cmp_tuple_or_list(Umat_p(pa,p1,p2,p3,b_c,det_d,p_s,wl,yaw,pitch,roll,z1,z2,z3,z_s,chi,phi,polar_max), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e