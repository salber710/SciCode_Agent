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
    
    # Calculate the reciprocal lattice vectors magnitude
    v_star = 2 * np.pi / (a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    ))
    
    # Calculate direction cosines
    ca = np.cos(alpha_rad)
    cb = np.cos(beta_rad)
    cg = np.cos(gamma_rad)
    sg = np.sin(gamma_rad)
    
    # Calculate components of B matrix using an alternative method
    B11 = v_star * b * c * sg
    B12 = v_star * c * (ca - cb * cg) / sg
    B13 = v_star * b * (cb - ca * cg) / sg
    B21 = 0
    B22 = v_star * c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg) / sg
    B23 = 0
    B31 = 0
    B32 = 0
    B33 = v_star * a * b * sg / np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    
    # Construct the B matrix
    B = np.array([
        [B11, B12, B13],
        [B21, B22, B23],
        [B31, B32, B33]
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