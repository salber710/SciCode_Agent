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


    # Calculate the wave vector magnitude for the given wavelength
    k = 1 / wl

    # Calculate the pixel offsets from the beam center in terms of physical distance
    offset_x_mm = (p[0] - b_c[0]) * p_s
    offset_y_mm = (p[1] - b_c[1]) * p_s

    # Compute the effective distance from the sample to the detector pixel using the Pythagorean theorem
    effective_distance = np.sqrt(det_d**2 + offset_x_mm**2 + offset_y_mm**2)

    # Determine the scattered wave vector in the lab coordinate system
    k_s_x_lab = k * det_d / effective_distance
    k_s_y_lab = -k * offset_x_mm / effective_distance
    k_s_z_lab = -k * offset_y_mm / effective_distance

    # Create the scattered wave vector array in lab coordinates
    k_s = np.array([k_s_x_lab, k_s_y_lab, k_s_z_lab])

    # Define the incident wave vector aligned along the x-axis in the lab frame
    k_i = np.array([k, 0, 0])

    # Compute the momentum transfer vector Q by subtracting the incident wave vector from the scattered wave vector
    Q = k_s - k_i

    # Return the momentum transfer vector Q as a 3x1 column matrix
    return np.array([[Q[0]], [Q[1]], [Q[2]]])


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e