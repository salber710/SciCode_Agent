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
    Input
    p: detector pixel (x,y), a tuple of two integers
    b_c: incident beam center at detector pixel (xc,yc), a tuple of floats
    det_d: sample distance to the detector, float in the unit of mm
    p_s: detector pixel size, and each pixel is a square, float in the unit of mm
    wl: X-ray wavelength, float in the unit of angstrom
    Output
    Q: a 3x1 matrix, float in the unit of inverse angstrom
    '''

    # Unpack pixel coordinates
    x_det, y_det = p
    x_c, y_c = b_c

    # Calculate the position of the pixel in the detector plane
    x_pos = (x_det - x_c) * p_s
    y_pos = (y_det - y_c) * p_s

    # Calculate the scattered beam vector k_s
    # The detector plane is perpendicular to the incident beam, so the z component is det_d
    k_s = np.array([x_pos, y_pos, det_d])
    k_s_magnitude = np.linalg.norm(k_s)
    
    # Normalize k_s to have a magnitude of 1/wl
    k_s = (1/wl) * (k_s / k_s_magnitude)

    # Calculate the incident beam vector k_i
    k_i = np.array([1/wl, 0, 0])  # Along the x direction

    # Calculate the momentum transfer Q
    Q = k_s - k_i

    return Q


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