import numpy as np
import numpy.linalg as la

# Background: Graphene is a two-dimensional material consisting of carbon atoms arranged in a hexagonal lattice. 
# The lattice can be described using two basis vectors. In graphene, the armchair direction is typically along the y-axis, 
# and the zigzag direction is along the x-axis. The lattice constant 'a' is the distance between two adjacent carbon atoms. 
# In a monolayer graphene sheet, each carbon atom is bonded to three other carbon atoms, forming a hexagonal pattern. 
# The task is to generate the coordinates of the carbon atoms in a monolayer graphene sheet, considering a sliding distance 's' 
# in the y-direction, a z-coordinate 'z', and a supercell size 'n' which determines the number of lattice sites in both 
# positive and negative x and y directions.


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
    if not isinstance(n, int):
        raise TypeError("Supercell size 'n' must be an integer.")
    if n < 0:
        raise ValueError("Supercell size 'n' must be non-negative.")
    if not all(isinstance(i, (int, float)) for i in [s, a, z]):
        raise TypeError("Inputs s, a, and z must be numeric.")

    # Define the basis vectors for the graphene lattice
    a1 = np.array([np.sqrt(3) * a, 0])  # Zigzag direction
    a2 = np.array([np.sqrt(3) * a / 2, 3 * a / 2])  # Armchair direction

    # Define the basis atoms in the unit cell
    basis_atoms = [
        np.array([0, 0]),
        np.array([np.sqrt(3) * a / 2, a / 2])
    ]

    # Initialize a list to store the atom positions
    atom_positions = []

    # Loop over the supercell range
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the position of each atom in the supercell
            for basis in basis_atoms:
                position = i * a1 + j * a2 + basis
                # Apply the sliding distance in the y-direction
                position[1] += s
                # Append the position with the z-coordinate
                atom_positions.append([position[0], position[1], z])

    # Convert the list to a numpy array
    atoms = np.array(atom_positions)

    return atoms



# Background: In computational materials science, the normal vector at a point on a surface is a vector that is perpendicular to the surface at that point. For a graphene sheet, each carbon atom is bonded to three nearest neighbors, forming a hexagonal lattice. The normal vector at each atom can be determined by averaging the cross products of vectors to its three nearest neighbors. The cross product of two vectors results in a vector that is perpendicular to the plane containing the two vectors. By normalizing these cross products and averaging them, we obtain a normal vector for each atom. The direction of the normal vector is important; it should point outward from the surface. For atoms with a positive z-coordinate, the normal should point in the negative z-direction, and vice versa. If the calculated normal vector points in the wrong direction, it can be corrected by multiplying it by -1.



def assign_normals(xyzs):
    '''Assign normal vectors on the given atoms
    Args:
        xyzs (np.array): Shape (natoms, 3)
    Returns:
        normed_cross_avg (np.array): Shape (natoms, 3)
    '''
    natoms = xyzs.shape[0]
    normed_cross_avg = np.zeros((natoms, 3))

    # Iterate over each atom to calculate its normal vector
    for i in range(natoms):
        # Find the 3 nearest neighbors
        distances = la.norm(xyzs - xyzs[i], axis=1)
        nearest_indices = np.argsort(distances)[1:4]  # Exclude the atom itself

        # Calculate the cross products of vectors to the 3 nearest neighbors
        cross_products = []
        for j in range(3):
            v1 = xyzs[nearest_indices[j]] - xyzs[i]
            v2 = xyzs[nearest_indices[(j + 1) % 3]] - xyzs[i]
            cross_product = np.cross(v1, v2)
            cross_products.append(cross_product / la.norm(cross_product))

        # Average the normalized cross products
        normal = np.mean(cross_products, axis=0)

        # Normalize the averaged vector
        normal /= la.norm(normal)

        # Ensure the normal vector points in the correct z-direction
        if (xyzs[i, 2] > 0 and normal[2] > 0) or (xyzs[i, 2] < 0 and normal[2] < 0):
            normal = -normal

        normed_cross_avg[i] = normal

    return normed_cross_avg

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('66.2', 3)
target = targets[0]

assert np.allclose(assign_normals(generate_monolayer_graphene(0, 2.46, 1.8, 1)), target)
target = targets[1]

assert np.allclose(assign_normals(generate_monolayer_graphene(0, 2.46, -1.8, 1)), target)
target = targets[2]

assert np.allclose(assign_normals(generate_monolayer_graphene((-2/3)*3**0.5*2.46, 2.46, -1.8, 1)), target)
