import numpy as np



# Background: In crystallography, the reciprocal lattice is a conceptual lattice used to understand various properties of the crystal structure. 
# The transformation from reciprocal lattice coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) involves a matrix known as the B matrix.
# The B matrix is derived from the direct lattice parameters (a, b, c, alpha, beta, gamma), where a, b, and c are the lengths of the unit cell edges,
# and alpha, beta, gamma are the angles between these edges. The reciprocal lattice vectors are defined such that they are orthogonal to the direct lattice vectors.
# The transformation matrix B is constructed using the metric tensor of the direct lattice, which is calculated from the lattice parameters.
# The metric tensor G is a 3x3 matrix that contains the dot products of the direct lattice vectors. The B matrix is then derived from the inverse of this metric tensor.


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
    
    # Calculate the reciprocal lattice parameters
    a_star = b * c * np.sin(alpha_rad) / volume
    b_star = a * c * np.sin(beta_rad) / volume
    c_star = a * b * np.sin(gamma_rad) / volume
    
    # Calculate the cosines of the angles between the reciprocal lattice vectors
    cos_alpha_star = (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad)) / (np.sin(beta_rad) * np.sin(gamma_rad))
    cos_beta_star = (np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad)) / (np.sin(alpha_rad) * np.sin(gamma_rad))
    cos_gamma_star = (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad)) / (np.sin(alpha_rad) * np.sin(beta_rad))
    
    # Calculate the sines of the angles between the reciprocal lattice vectors
    sin_gamma_star = np.sqrt(1 - cos_gamma_star**2)
    
    # Construct the B matrix
    B = np.array([
        [a_star, b_star * cos_gamma_star, c_star * cos_beta_star],
        [0, b_star * sin_gamma_star, c_star * (cos_alpha_star - cos_beta_star * cos_gamma_star) / sin_gamma_star],
        [0, 0, c_star * volume / (a * b * sin_gamma_star)]
    ])
    
    return B

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('73.1', 3)
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
