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
    # Lattice vectors for a different basis
    basis_vectors = np.array([
        [a * 3 / 2, np.sqrt(3) * a / 2],
        [-a * 3 / 2, np.sqrt(3) * a / 2]
    ])
    
    # Define the unit cell with a unique arrangement
    unit_cell_atoms = np.array([
        [0, 0],                  # Atom A at origin
        [a * 3 / 4, np.sqrt(3) * a / 4]  # Atom B, offset in a unique way
    ])
    
    # List to store positions of all atoms
    all_atoms = []

    # Generate atomic positions within the supercell grid
    for u in range(-n, n + 1):
        for v in range(-n, n + 1):
            # Calculate translation vector using the defined basis
            translation = u * basis_vectors[0] + v * basis_vectors[1]
            # Incorporate sliding distance in y-direction
            translation[1] += s
            # Calculate full atomic positions and store them
            for atom in unit_cell_atoms:
                atom_position = atom + translation
                all_atoms.append([atom_position[0], atom_position[1], z])
    
    return np.array(all_atoms)


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