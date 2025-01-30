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
    # Define the transformation matrix for the graphene lattice as a combination of a1 and a2
    transform_matrix = np.array([[a, a / 2], [0, a * np.sqrt(3) / 2]])
    
    # Define the atomic positions in the unit cell
    unit_cell_positions = np.array([
        [0, 0],  # Atom A at origin
        [1/2, 1/2]  # Atom B in fractional coordinates
    ])

    # Prepare to store the full list of atomic positions
    atom_positions = []

    # Loop through each possible supercell translation
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the translation vector using the transformation matrix
            translation_vector = np.dot(transform_matrix, [i, j])
            
            # Apply the sliding distance to the y-coordinate
            translation_vector[1] += s
            
            # Generate full atomic positions by adding translation to each unit cell position
            for atom in unit_cell_positions:
                real_position = np.dot(transform_matrix, atom) + translation_vector
                atom_positions.append([real_position[0], real_position[1], z])

    # Convert list of positions to a numpy array
    atoms = np.array(atom_positions)
    
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