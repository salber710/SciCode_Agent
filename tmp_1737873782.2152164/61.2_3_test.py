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

    # Unpack inputs
    x, y = p
    xc, yc = b_c

    # Calculate the position of the pixel in the detector coordinate system
    x_det = (x - xc) * p_s  # in mm
    y_det = (y - yc) * p_s  # in mm

    # Calculate the incident wavevector ki (in the direction of +x in lab coordinates)
    k_magnitude = 2 * np.pi / wl  # in inverse angstroms
    k_i = np.array([k_magnitude, 0, 0])  # ki is along the +x direction

    # Calculate the scattered wavevector ks
    # Convert detector coordinates to lab coordinates
    x_lab = -y_det  # in lab coordinates, x_det aligns with -y in detector coordinates
    y_lab = -x_det  # in lab coordinates, y_det aligns with -z in detector coordinates
    z_lab = det_d  # z_lab is the sample to detector distance

    # Calculate the magnitude of the scattered wavevector ks
    ks_magnitude = k_magnitude

    # Normalize the scattered direction vector
    norm = np.sqrt(x_lab**2 + y_lab**2 + z_lab**2)
    scattered_direction = np.array([x_lab, y_lab, z_lab]) / norm

    # Calculate ks vector
    k_s = ks_magnitude * scattered_direction  # in inverse angstroms

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