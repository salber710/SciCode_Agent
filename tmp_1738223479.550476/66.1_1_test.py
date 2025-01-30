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
    # Define constants based on graphene geometry
    sqrt3 = np.sqrt(3)
    # Define primitive vectors for graphene lattice with armchair direction along y-axis
    b1 = np.array([a / 2, a * sqrt3 / 2])
    b2 = np.array([a, 0])
    
    # Define positions of atoms in the unit cell
    unit_cell_positions = np.array([
        [0, 0],          # Atom A
        [a / 2, a * sqrt3 / 2]  # Atom B
    ])
    
    # Initialize list to collect atom coordinates
    atom_coordinates = []
    
    # Iterate over each supercell in specified range
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # Calculate translation vector for current cell
            translation_vector = i * b1 + j * b2
            # Adjust y-component with sliding distance s
            translation_vector[1] += s
            # Add atoms to list with translation and fixed z-coordinate
            for pos in unit_cell_positions:
                new_pos = pos + translation_vector
                atom_coordinates.append([new_pos[0], new_pos[1], z])
    
    # Convert list to numpy array
    atoms = np.array(atom_coordinates)
    
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