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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('66.1', 3)
target = targets[0]

s=0
a=2.46
z=0
n=1
assert np.allclose(generate_monolayer_graphene(s, a, z, n), target)
target = targets[1]

s=0
a=2.46
z=1.7
n=1
assert np.allclose(generate_monolayer_graphene(s, a, z, n), target)
target = targets[2]

s=(-2/3)*3**0.5*2.46
a=2.46
z=0
n=1
assert np.allclose(generate_monolayer_graphene(s, a, z, n), target)
