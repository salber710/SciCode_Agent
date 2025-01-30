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

    # Unpack the direct lattice parameters
    a, b, c, alpha, beta, gamma = pa

    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate the volume of the unit cell
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 +
        2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Calculate components of the reciprocal lattice vectors
    bx = 2 * np.pi * b * c * np.sin(alpha_rad) / volume
    by = 2 * np.pi * a * c * np.sin(beta_rad) / volume
    bz = 2 * np.pi * a * b * np.sin(gamma_rad) / volume

    # The B matrix is defined as:
    B = np.array([
        [bx, 0, 0],
        [by * np.cos(gamma_rad), by * np.sin(gamma_rad), 0],
        [bz * np.cos(beta_rad), bz * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad), bz * np.sqrt(1 - (np.cos(alpha_rad)**2 + (np.cos(beta_rad) - np.cos(alpha_rad) * np.cos(gamma_rad))**2) / np.sin(gamma_rad)**2)]
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