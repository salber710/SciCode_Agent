from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: 
# In crystallography, the reciprocal lattice is a conceptual lattice used to understand the diffraction patterns of crystals. 
# The reciprocal lattice vectors are defined in terms of the direct lattice vectors. If we have direct lattice vectors 
# a, b, and c, the reciprocal lattice vectors a*, b*, and c* are defined such that:
# a* = (b x c) / (a . (b x c))
# b* = (c x a) / (b . (c x a))
# c* = (a x b) / (c . (a x b))
# Here, x denotes the cross product and . denotes the dot product.
# The transformation from reciprocal lattice coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) 
# involves the use of a transformation matrix B. This matrix is derived from the reciprocal lattice vectors 
# expressed in Cartesian coordinates.
# The angles alpha, beta, and gamma are the angles between the lattice vectors. 
# They are used to calculate the volume of the unit cell and the components of the reciprocal lattice vectors.


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
    
    # Calculate the components of the reciprocal lattice vectors
    a_star = (b * c * np.sin(alpha_rad)) / volume
    b_star = (c * a * np.sin(beta_rad)) / volume
    c_star = (a * b * np.sin(gamma_rad)) / volume
    
    # Calculate the B matrix
    B = np.array([
        [a_star, b_star * np.cos(gamma_rad), c_star * np.cos(beta_rad)],
        [0, b_star * np.sin(gamma_rad), -c_star * np.sin(beta_rad) * np.cos(alpha_rad)],
        [0, 0, c_star * np.sin(beta_rad) * np.sin(alpha_rad)]
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