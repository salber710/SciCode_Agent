from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import numpy.linalg as la


def generate_monolayer_graphene(s, a, z, n):
    '''Generate the geometry of monolayer graphene.
    Args:
        s (float): Horizontal in-plane sliding distance.
        a (float): Lattice constant.
        z (float): z-coordinate
        n (int): supercell size
    Returns:
        atoms (np.array): Array containing the x, y, and z coordinates of the atoms.
    '''
    # Define a new set of lattice vectors
    lattice_vectors = np.array([
        [a, a * np.sqrt(3) / 3],
        [-a / 2, a * np.sqrt(3) / 3]
    ])
    
    # Define unit cell atoms with a different configuration
    unit_cell = np.array([
        [0, 0],  # Atom A
        [a / 4, a * np.sqrt(3) / 6]  # Atom B
    ])

    # Initialize the list to collect the atom positions
    atom_positions = []

    # Iterate over the grid to construct the supercell
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the translation vector for the current grid point
            translation = i * lattice_vectors[0] + j * lattice_vectors[1]
            # Apply the sliding distance to the y-component
            translation[1] += s
            # Calculate the positions of the atoms in the translated unit cell
            for atom in unit_cell:
                atom_pos = atom + translation
                atom_positions.append([atom_pos[0], atom_pos[1], z])

    # Convert list to a numpy array before returning
    atoms = np.array(atom_positions)
    
    return atoms



def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''
    natoms = xyzs.shape[0]
    normed_cross_avg = np.zeros((natoms, 3))
    
    # Precompute pairwise distance matrix using squared distances for efficiency
    dist_squared = np.sum((xyzs[:, np.newaxis, :] - xyzs[np.newaxis, :, :]) ** 2, axis=2)

    for i in range(natoms):
        # Identify indices of the three nearest neighbors excluding the atom itself
        neighbors_indices = np.argpartition(dist_squared[i], 4)[:4]
        neighbors_indices = neighbors_indices[neighbors_indices != i][:3]

        # Calculate vectors to each of the three neighbors
        vectors = xyzs[neighbors_indices] - xyzs[i]

        # Compute cross products by cyclically rotating indices for a different combination
        cross_01 = np.cross(vectors[0], vectors[2])
        cross_12 = np.cross(vectors[1], vectors[0])
        cross_20 = np.cross(vectors[2], vectors[1])

        # Use an arithmetic mean of cross products as the normal vector
        cross_mean = (cross_01 + cross_12 + cross_20) / 3

        # Normalize the resulting vector
        normed_cross = cross_mean / np.linalg.norm(cross_mean)

        # Ensure the normal vector points in the correct z-direction
        if (xyzs[i][2] > 0 and normed_cross[2] > 0) or (xyzs[i][2] < 0 and normed_cross[2] < 0):
            normed_cross *= -1
        
        normed_cross_avg[i] = normed_cross

    return normed_cross_avg



def potential_repulsive(r_ij, n_i, n_j, z0, C, C0, C2, C4, delta, lamda):
    '''Define repulsive potential.
    Args:
        r_ij: (nmask, 3)
        n_i: (nmask, 3)
        n_j: (nmask, 3)
        z0 (float): KC parameter
        C (float): KC parameter
        C0 (float): KC parameter
        C2 (float): KC parameter
        C4 (float): KC parameter
        delta (float): KC parameter
        lamda (float): KC parameter
    Returns:
        pot (nmask): values of repulsive potential for the given atom pairs.
    '''

    # Calculate r_ij magnitudes using np.sqrt and einsum
    r_ij_magnitude = np.sqrt(np.einsum('ij,ij->i', r_ij, r_ij))

    # Calculate dot products with np.inner
    r_dot_n_i = np.inner(r_ij, n_i)
    r_dot_n_j = np.inner(r_ij, n_j)

    # Calculate rho squared values using np.clip to ensure non-negativity
    rho_ij_squared = np.clip(r_ij_magnitude**2 - r_dot_n_i**2, 0, None)
    rho_ji_squared = np.clip(r_ij_magnitude**2 - r_dot_n_j**2, 0, None)

    # Define f(rho) using a vectorized approach
    def f(rho_squared):
        x = rho_squared / delta**2
        return np.exp(-x) * (C0 + C2 * x + C4 * x**2)

    # Calculate f(rho) for rho_ij and rho_ji
    f_rho_ij = f(rho_ij_squared)
    f_rho_ji = f(rho_ji_squared)

    # Calculate the exponential part of the potential
    exp_component = np.exp(-lamda * (r_ij_magnitude - z0)) * (C + f_rho_ij + f_rho_ji)

    # Calculate the attractive component using np.divide
    attractive_component = np.divide(1, np.power(r_ij_magnitude / z0, 6), where=r_ij_magnitude!=0)

    # Calculate the repulsive potential
    pot = exp_component - attractive_component

    return pot




def potential_attractive(rnorm, z0, A, lambda_, C, f):
    '''Define attractive potential.
    Args:
        rnorm (float or np.array): distance
        z0 (float): KC parameter
        A (float): KC parameter
        lambda_ (float): exponential decay parameter
        C (float): constant term in the exponential component
        f (callable): function taking density arguments
    Returns:
        pot (float): calculated potential
    '''
    # Calculate the exponential component in a new form
    exp_component = np.exp(lambda_ * (z0 - rnorm))
    
    # Use a different transformation of f arguments
    f_term_1 = f(rnorm + z0)
    f_term_2 = f(abs(rnorm - z0))
    
    # Compute the attractive potential with a unique combination of terms
    pot = exp_component * (C * f_term_1 - f_term_2) - A * (z0 / rnorm) ** 6
    
    return pot


try:
    targets = process_hdf5_to_tuple('66.4', 3)
    target = targets[0]
    z0 = 3.370060885645178
    C0 = 21.783338516870739
    C2 = 10.469388694543325
    C4 = 8.864962486046355
    C = 0.000013157376477
    delta = 0.723952360283636
    lamda = 3.283145920221462
    A = 13.090159601618883
    n_i = np.array([[0, 0, -1]])
    n_j = np.array([[0, 0, 1]])
    assert np.allclose(potential_attractive(3.2, z0, A), target)

    target = targets[1]
    z0 = 3.370060885645178
    C0 = 21.783338516870739
    C2 = 10.469388694543325
    C4 = 8.864962486046355
    C = 0.000013157376477
    delta = 0.723952360283636
    lamda = 3.283145920221462
    A = 13.090159601618883
    n_i = np.array([[0, 0, -1]])
    n_j = np.array([[0, 0, 1]])
    assert np.allclose(potential_attractive(4.03628542 , z0, A), target)

    target = targets[2]
    z0 = 3.370060885645178
    C0 = 21.783338516870739
    C2 = 10.469388694543325
    C4 = 8.864962486046355
    C = 0.000013157376477
    delta = 0.723952360283636
    lamda = 3.283145920221462
    A = 13.090159601618883
    n_i = np.array([[0, 0, -1]])
    n_j = np.array([[0, 0, 1]])
    assert np.allclose(potential_attractive(5.86910555, z0, A), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e