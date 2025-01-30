import numpy as np



# Background: In crystallography, the reciprocal lattice is a construct used to understand the diffraction patterns of crystals. 
# The reciprocal lattice vectors are related to the direct lattice vectors through the metric tensor. 
# The transformation from reciprocal lattice coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) involves the B matrix.
# The B matrix is derived from the direct lattice parameters (a, b, c, alpha, beta, gamma), where alpha, beta, and gamma are the angles between the lattice vectors.
# The transformation is based on the relationships between the direct and reciprocal lattice vectors, where the reciprocal lattice vectors are defined as:
# b1 = 2π * (b × c) / (a · (b × c))
# b2 = 2π * (c × a) / (b · (c × a))
# b3 = 2π * (a × b) / (c · (a × b))
# The B matrix is constructed using these reciprocal lattice vectors, and it transforms (h, k, l) to Cartesian coordinates.


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
    volume = a * b * c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 
                                 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
    
    # Calculate the reciprocal lattice vectors
    b1 = 2 * np.pi * b * c * np.sin(alpha_rad) / volume
    b2 = 2 * np.pi * c * a * np.sin(beta_rad) / volume
    b3 = 2 * np.pi * a * b * np.sin(gamma_rad) / volume
    
    # Calculate the components of the B matrix
    B = np.zeros((3, 3))
    B[0, 0] = b1
    B[1, 1] = b2
    B[2, 2] = b3
    
    # Return the B matrix
    return B

from scicode.parse.parse import process_hdf5_to_tuple
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
