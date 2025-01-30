from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import numpy.linalg as la



# Background: Graphene is a single layer of carbon atoms arranged in a two-dimensional honeycomb lattice.
# The lattice has two distinct directions: the armchair direction and the zigzag direction. In this problem,
# the armchair direction is along the y-axis, and the zigzag direction is along the x-axis. The basic unit
# of the graphene lattice can be described using two lattice vectors. The lattice constant `a` defines the
# distance between adjacent carbon atoms along the zigzag direction. For graphene, the distance between
# nearest neighbors is typically 1.42 Å. The sliding distance `s` represents a shift of the entire lattice
# in the y-direction. This is used to simulate variations in the lattice position. The z-coordinate `z` is
# constant for all atoms in a monolayer graphene plane. The parameter `n` defines the number of repeating
# units (supercells) in both the positive and negative x and y directions from the origin.


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
    sqrt3 = np.sqrt(3)
    # Define the primitive lattice vectors in the x and y plane
    a1 = np.array([a, 0])
    a2 = np.array([a / 2, a * sqrt3 / 2])
    
    # Generate positions for each atom in the unit cell
    unit_cell = np.array([
        [0, 0],                # Atom A
        [a / 2, a * sqrt3 / 2] # Atom B
    ])
    
    # Create a list to store all the atom coordinates
    atoms = []
    
    # Iterate over the range of supercells in both x and y directions
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the translation vector for the current supercell
            translation = i * a1 + j * a2
            # Adjust the y-component by the sliding distance s
            translation[1] += s
            # Add the atoms of the unit cell to the list, including the translation and z-coordinate
            for atom in unit_cell:
                position = atom + translation
                atoms.append([position[0], position[1], z])
    
    # Convert the list of atoms to a NumPy array
    atoms = np.array(atoms)
    
    return atoms


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e