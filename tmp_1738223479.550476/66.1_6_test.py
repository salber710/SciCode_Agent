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
    # Alternative approach using direct coordinate calculations
    # Define lattice vectors differently.
    armchair_vector = np.array([a * 3/2, np.sqrt(3) * a / 2])
    zigzag_vector = np.array([a * 3/2, -np.sqrt(3) * a / 2])
    
    # Define the unit cell with custom coordinates
    unit_cell = np.array([
        [0, 0],  # Atom A position
        [a, np.sqrt(3) * a / 3]  # Atom B position
    ])
    
    # Create a list to hold the atom positions
    atom_list = []
    
    # Iterate over grid defined by n
    for p in range(-n, n + 1):
        for q in range(-n, n + 1):
            # Calculate translation vector based on armchair and zigzag vectors
            translation = p * armchair_vector + q * zigzag_vector
            # Add sliding distance to the y-component
            translation[1] += s
            
            # Calculate atom positions in translated unit cell
            for atom in unit_cell:
                atom_position = atom + translation
                # Add z-coordinate and append to the list
                atom_list.append([atom_position[0], atom_position[1], z])
    
    # Convert the list of atom positions to a numpy array
    atoms = np.array(atom_list)
    
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