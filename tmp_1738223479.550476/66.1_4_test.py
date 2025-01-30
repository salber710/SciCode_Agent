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
    # Define the lattice vectors in a different form
    lattice_vectors = np.array([
        [3 * a / 2, np.sqrt(3) * a / 2],
        [3 * a / 2, -np.sqrt(3) * a / 2]
    ])
    
    # Define atomic positions in unit cell with a different approach
    unit_cell_positions = np.array([
        [0, 0],                          # Atom A
        [a * 3 / 2, np.sqrt(3) * a / 2]  # Atom B, adjusted for clarity
    ])
    
    # Initialize the list to collect atom coordinates
    atom_positions = []

    # Loop through indices in the range to cover the supercell
    for x_idx in range(-n, n + 1):
        for y_idx in range(-n, n + 1):
            # Calculate the translation vector for the current grid point
            translation = (x_idx * lattice_vectors[0] + y_idx * lattice_vectors[1])
            # Apply sliding distance to the y component
            translation[1] += s
            # Generate atomic positions by applying the translation
            for atom in unit_cell_positions:
                atom_pos = atom + translation
                atom_positions.append([atom_pos[0], atom_pos[1], z])

    # Convert to numpy array for the final output
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