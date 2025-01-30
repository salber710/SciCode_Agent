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



# Background: In crystallography, the momentum transfer vector \( \vec{Q} \) is a key concept used to describe the change in momentum of a photon due to scattering. It is defined as the difference between the wave vectors of the scattered (\( \vec{k_s} \)) and incident beams (\( \vec{k_i} \)). In this task, we need to calculate \( \vec{Q} \) at a specific detector pixel using the given coordinate systems. The incident beam direction is aligned with the positive x-axis, and the detector plane is perpendicular to it. The detector's coordinate system is rotated such that \( \mathbf{\hat{x}}_{det} \) is aligned with the negative y-axis and \( \mathbf{\hat{y}}_{det} \) is aligned with the negative z-axis. The wave vector magnitude \( k \) is given by \( k = 1/\lambda \), where \( \lambda \) is the wavelength.

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
    # Convert wavelength to wave vector k
    k = 1 / wl  # wave vector magnitude

    # Calculate the displacement of the pixel from the beam center in detector coordinates
    dx_det = (p[0] - b_c[0]) * p_s  # in mm
    dy_det = (p[1] - b_c[1]) * p_s  # in mm

    # Calculate the direction of the scattered wave vector in the detector coordinates
    # The detector is assumed to be perpendicular to the incident beam, so:
    kx_s = k * det_d / np.sqrt(det_d**2 + dx_det**2 + dy_det**2)
    ky_s = -k * dx_det / np.sqrt(det_d**2 + dx_det**2 + dy_det**2)
    kz_s = -k * dy_det / np.sqrt(det_d**2 + dx_det**2 + dy_det**2)

    # Calculate the scattered wave vector in the lab coordinates
    # The lab coordinates align with the incident beam direction
    k_s = np.array([kx_s, ky_s, kz_s])

    # The incident wave vector in lab coordinates is along the x-direction
    k_i = np.array([k, 0, 0])

    # Calculate the momentum transfer vector Q
    Q = k_s - k_i

    # Return Q as a 3x1 matrix
    return Q.reshape(3, 1)


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