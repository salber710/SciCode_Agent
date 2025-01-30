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
    
    # Calculate the volume of the direct lattice unit cell
    V = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )
    
    # Calculate the sine and cosine values
    sin_alpha = np.sin(alpha_rad)
    sin_beta = np.sin(beta_rad)
    sin_gamma = np.sin(gamma_rad)
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    
    # Pre-calculate auxiliary terms
    sigma = (cos_alpha - cos_beta * cos_gamma) / sin_beta
    
    # Calculate the components of the B matrix
    B11 = 2 * np.pi * b * c * sin_alpha / V
    B12 = -2 * np.pi * b * c * sigma / (V * sin_gamma)
    B13 = 2 * np.pi * (b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)) / V
    B22 = 2 * np.pi * c * a * sin_beta / V
    B23 = -2 * np.pi * c * a * (cos_beta - cos_alpha * cos_gamma) / (V * sin_gamma)
    B33 = 2 * np.pi * a * b * sin_gamma / V
    
    # Construct the B matrix
    B = np.array([
        [B11, B12, B13],
        [0,   B22, B23],
        [0,   0,   B33]
    ])
    
    return B


try:
    targets = process_hdf5_to_tuple('61.1', 3)
    target = targets[0]
    a,b,c,alpha,beta,gamma = (5.39097,5.39097,5.39097,89.8,90.1,89.5)
    pa = (a,b,c,alpha,beta,gamma)
    assert np.allclose(Bmat(pa), target)

    target = targets[1]
    a,b,c,alpha,beta,gamma = (5.41781,5.41781,5.41781,89.8,90.1,89.5)
    pa = (a,b,c,alpha,beta,gamma)
    assert np.allclose(Bmat(pa), target)

    target = targets[2]
    a,b,c,alpha,beta,gamma = (3.53953,3.53953,6.0082,89.8,90.1,120.1)
    pa = (a,b,c,alpha,beta,gamma)
    assert np.allclose(Bmat(pa), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e