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
    # Hexagonal lattice vectors, different representation
    v1 = np.array([0.5 * a, np.sqrt(3) * a / 2])
    v2 = np.array([-0.5 * a, np.sqrt(3) * a / 2])
    
    # Unit cell configuration with different ordering
    unit_cell = np.array([
        [0, 0],  # Atom A
        [0.5 * a, np.sqrt(3) * a / 6]  # Atom B, slightly different y-offset
    ])

    atoms = []

    # Loop through cells, using a different approach to arrange indices
    for m in range(-n, n + 1):
        for n in range(-n, n + 1):
            # Calculate translation
            trans = m * v1 + n * v2
            # Apply sliding distance
            trans[1] += s
            # Append translated unit cell atoms
            for pos in unit_cell:
                new_pos = pos + trans
                atoms.append([new_pos[0], new_pos[1], z])
    
    return np.array(atoms)


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