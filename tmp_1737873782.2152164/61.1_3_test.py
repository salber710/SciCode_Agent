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

    # Calculate volume of the unit cell
    V = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Calculate the reciprocal lattice vectors
    a_star = (b * c * np.sin(alpha_rad)) / V
    b_star = (a * c * np.sin(beta_rad)) / V
    c_star = (a * b * np.sin(gamma_rad)) / V

    # Calculate components of the B matrix
    B = np.zeros((3, 3))
    B[0, 0] = a_star
    B[0, 1] = b_star * np.cos(gamma_rad)
    B[0, 2] = c_star * np.cos(beta_rad)
    B[1, 1] = b_star * np.sin(gamma_rad)
    B[1, 2] = (c_star * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))) / np.sin(gamma_rad)
    B[2, 2] = V / (a * b * np.sin(gamma_rad))

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