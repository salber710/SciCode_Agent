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
    
    # Calculate the volume of the unit cell
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )
    
    # Calculate the components of the reciprocal lattice vectors in Cartesian coordinates
    V_star = (2 * np.pi) / volume
    a_star_x = V_star * b * c * np.sin(alpha_rad)
    a_star_y = 0
    a_star_z = 0
    b_star_x = V_star * c * (np.cos(beta_rad) - np.cos(alpha_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    b_star_y = V_star * c * a * np.sin(gamma_rad)
    b_star_z = 0
    c_star_x = V_star * a * b * (np.cos(gamma_rad) - np.cos(alpha_rad) * np.cos(beta_rad)) / np.sin(beta_rad)
    c_star_y = V_star * a * b * ((np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / (np.sin(beta_rad) * np.sin(gamma_rad)))
    c_star_z = V_star * a * b * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(beta_rad)

    # Construct the B matrix
    B = np.array([
        [a_star_x, b_star_x, c_star_x],
        [a_star_y, b_star_y, c_star_y],
        [a_star_z, b_star_z, c_star_z]
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