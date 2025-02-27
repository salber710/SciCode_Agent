import numpy as np
import numpy.linalg as la

# Background: Graphene is a two-dimensional material consisting of carbon atoms arranged in a hexagonal lattice. 
# The lattice can be described using two basis vectors. In the case of graphene, the armchair direction is along 
# the y-axis, and the zigzag direction is along the x-axis. The lattice constant 'a' is the distance between 
# adjacent carbon atoms. The geometry of monolayer graphene can be generated by placing atoms at specific 
# coordinates based on these lattice vectors. The sliding distance 's' in the y-direction allows for the 
# simulation of a shifted lattice, which is useful in studying bilayer graphene and other phenomena. The 
# parameter 'n' determines the size of the supercell, which is the number of lattice sites to generate in 
# both positive and negative directions along the x and y axes.


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
    # Define the basis vectors for graphene
    a1 = np.array([np.sqrt(3) * a, 0])  # Zigzag direction
    a2 = np.array([np.sqrt(3) * a / 2, 3 * a / 2])  # Armchair direction

    # Initialize a list to store the atom positions
    atoms = []

    # Loop over the range to generate the supercell
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the position of the first atom in the unit cell
            pos1 = i * a1 + j * a2
            # Calculate the position of the second atom in the unit cell
            pos2 = pos1 + np.array([0, a])
            
            # Add the z-coordinate and sliding distance to the y-coordinate
            atoms.append([pos1[0], pos1[1] + s, z])
            atoms.append([pos2[0], pos2[1] + s, z])

    # Convert the list to a numpy array
    atoms = np.array(atoms)

    return atoms



# Background: In computational materials science, the normal vector at an atom in a lattice is crucial for understanding surface properties and interactions. For a given atom, the normal vector can be determined by considering its nearest neighbors. The cross product of vectors from the atom to its neighbors gives a vector perpendicular to the plane formed by these vectors. By averaging the cross products of vectors to the three nearest neighbors and normalizing the result, we obtain a normal vector for the atom. This vector should be adjusted to point in the correct direction based on the atom's z-coordinate: negative z-direction for atoms with z > 0 and positive z-direction for atoms with z < 0.

def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''



    natoms = xyzs.shape[0]
    normed_cross_avg = np.zeros((natoms, 3))

    for i in range(natoms):
        # Find the 3 nearest neighbors
        distances = la.norm(xyzs - xyzs[i], axis=1)
        nearest_indices = np.argsort(distances)[1:4]  # Exclude the atom itself

        # Calculate cross products of vectors to nearest neighbors
        cross_products = []
        for j in range(3):
            for k in range(j + 1, 3):
                vec1 = xyzs[nearest_indices[j]] - xyzs[i]
                vec2 = xyzs[nearest_indices[k]] - xyzs[i]
                cross_product = np.cross(vec1, vec2)
                cross_products.append(cross_product)

        # Average the cross products
        cross_avg = np.mean(cross_products, axis=0)

        # Normalize the average cross product
        normed_cross_avg[i] = cross_avg / la.norm(cross_avg)

        # Ensure the normal vector points in the correct z-direction
        if (xyzs[i, 2] > 0 and normed_cross_avg[i, 2] > 0) or (xyzs[i, 2] < 0 and normed_cross_avg[i, 2] < 0):
            normed_cross_avg[i] = -normed_cross_avg[i]

    return normed_cross_avg


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('66.2', 3)
target = targets[0]

assert np.allclose(assign_normals(generate_monolayer_graphene(0, 2.46, 1.8, 1)), target)
target = targets[1]

assert np.allclose(assign_normals(generate_monolayer_graphene(0, 2.46, -1.8, 1)), target)
target = targets[2]

assert np.allclose(assign_normals(generate_monolayer_graphene((-2/3)*3**0.5*2.46, 2.46, -1.8, 1)), target)
