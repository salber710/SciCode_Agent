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
    # Use a new set of lattice vectors for graphene
    lattice_vector_x = np.array([a, 0])
    lattice_vector_y = np.array([a / 2, a * np.sqrt(3) / 2])
    
    # Define the atomic positions in the unit cell with different offsets
    unit_cell_atoms = np.array([
        [0, 0],  # Atom A
        [a / 3, a * np.sqrt(3) / 3]  # Atom B
    ])
    
    # Initialize a list to collect all atom positions
    atom_positions = []

    # Iterate over the grid to generate the supercell
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate the translation vector
            translation = i * lattice_vector_x + j * lattice_vector_y
            # Apply the sliding distance to the y-component
            translation[1] += s
            # Add the translated atoms to the list
            for atom in unit_cell_atoms:
                pos = atom + translation
                atom_positions.append([pos[0], pos[1], z])
    
    # Convert the list to a numpy array
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