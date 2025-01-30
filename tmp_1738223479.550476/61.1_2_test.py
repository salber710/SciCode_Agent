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
    
    # Calculate the cosine values
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    
    # Calculate the sine values
    sin_alpha = np.sin(alpha_rad)
    sin_beta = np.sin(beta_rad)
    sin_gamma = np.sin(gamma_rad)

    # Calculate the volume of the direct lattice unit cell
    V = a * b * c * np.sqrt(
        1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    
    # Calculate the components of the reciprocal lattice vectors
    a_star = 2 * np.pi * b * c * sin_alpha / V
    b_star = 2 * np.pi * c * a * sin_beta / V
    c_star = 2 * np.pi * a * b * sin_gamma / V
    
    # Construct the B matrix using a different method
    B = np.array([
        [a_star, b_star * cos_gamma, c_star * cos_beta],
        [0, b_star * sin_gamma, c_star * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
        [0, 0, c_star * V / (a * b * c * sin_gamma)]
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