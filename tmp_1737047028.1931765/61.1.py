import numpy as np



# Background: In crystallography, the reciprocal lattice is a construct used to understand the diffraction patterns of crystals. 
# The reciprocal lattice vectors are defined in terms of the direct lattice vectors. The transformation from reciprocal lattice 
# coordinates (h, k, l) to Cartesian coordinates (q_x, q_y, q_z) involves a matrix, B, which is derived from the direct lattice 
# parameters (a, b, c, alpha, beta, gamma). The direct lattice parameters describe the geometry of the unit cell in real space, 
# where a, b, and c are the lengths of the cell edges, and alpha, beta, and gamma are the angles between these edges. 
# The reciprocal lattice vectors are defined such that a_i · b_j = δ_ij, where δ_ij is the Kronecker delta. 
# The transformation matrix B is constructed using these parameters and is used to convert reciprocal lattice coordinates 
# to Cartesian coordinates. The angles alpha, beta, and gamma are converted from degrees to radians for trigonometric calculations.


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
    
    # Calculate the volume of the unit cell in reciprocal space
    V = a * b * c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 
                            + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
    
    # Calculate the reciprocal lattice parameters
    a_star = b * c * np.sin(alpha_rad) / V
    b_star = a * c * np.sin(beta_rad) / V
    c_star = a * b * np.sin(gamma_rad) / V
    
    # Calculate the components of the B matrix
    B = np.array([
        [a_star, b_star * np.cos(gamma_rad), c_star * np.cos(beta_rad)],
        [0, b_star * np.sin(gamma_rad), -c_star * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)],
        [0, 0, c_star * V / (a * b * np.sin(gamma_rad))]
    ])
    
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
