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



# Background: In molecular and computational geometry, normal vectors are often used to describe the orientation of surfaces or atoms in a lattice. For an atom in a graphene sheet, its normal vector can be determined by considering the geometry formed with its nearest neighbors. The cross product of vectors between an atom and its neighbors gives a vector perpendicular to the plane formed by those vectors. Averaging these vectors and normalizing the result provides a smoothed normal vector for the atom. In this task, we need to ensure that the normal vectors point consistently in the direction relative to the z-coordinate of the atom, flipping them if necessary.



def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms,)
    '''
    # Initialize an array to store the normal vectors
    normed_cross_avg = np.zeros((xyzs.shape[0], 3))
    
    # Iterate through each atom to calculate its normal vector
    for i, atom in enumerate(xyzs):
        # Find the three nearest neighbors
        distances = la.norm(xyzs - atom, axis=1)
        neighbors_indices = np.argsort(distances)[1:4]  # Exclude the atom itself
        
        # Calculate the vectors to each of the three neighbors
        vectors = xyzs[neighbors_indices] - atom
        
        # Calculate cross products of vectors to derive normal vectors
        cross_vectors = []
        for j in range(3):
            v1 = vectors[j]
            v2 = vectors[(j + 1) % 3]
            cross_product = np.cross(v1, v2)
            cross_vectors.append(cross_product)
        
        # Average the cross products
        cross_avg = np.mean(cross_vectors, axis=0)
        
        # Normalize the averaged cross product to get the normal vector
        normed_cross_avg[i] = cross_avg / la.norm(cross_avg)
        
        # Ensure the normal vector points in the correct z-direction
        if (atom[2] > 0 and normed_cross_avg[i][2] > 0) or (atom[2] < 0 and normed_cross_avg[i][2] < 0):
            normed_cross_avg[i] *= -1
    
    return normed_cross_avg


try:
    targets = process_hdf5_to_tuple('66.2', 3)
    target = targets[0]
    assert np.allclose(assign_normals(generate_monolayer_graphene(0, 2.46, 1.8, 1)), target)

    target = targets[1]
    assert np.allclose(assign_normals(generate_monolayer_graphene(0, 2.46, -1.8, 1)), target)

    target = targets[2]
    assert np.allclose(assign_normals(generate_monolayer_graphene((-2/3)*3**0.5*2.46, 2.46, -1.8, 1)), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e