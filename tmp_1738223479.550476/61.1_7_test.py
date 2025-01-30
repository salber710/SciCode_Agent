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
    alpha_rad, beta_rad, gamma_rad = np.radians([alpha, beta, gamma])
    
    # Calculate trigonometric functions
    ca, cb, cg = np.cos([alpha_rad, beta_rad, gamma_rad])
    sa, sb, sg = np.sin([alpha_rad, beta_rad, gamma_rad])
    
    # Volume of the direct lattice unit cell
    V = a * b * c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    v_star = 2 * np.pi / V
    
    # Construct the B matrix using a different approach
    B = np.array([
        [v_star * b * c * sa, v_star * c * (ca - cb * cg) / sg, v_star * (cb - ca * cg) / sb],
        [0, v_star * a * c * sb, v_star * (ca - cb * cg) / sa],
        [0, 0, v_star * a * b * sg]
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