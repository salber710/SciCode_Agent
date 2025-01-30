from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''


    # Ensure r is a numpy array
    r = np.array(r)

    # Calculate the wrapped coordinates using a combination of shifting and conditional checks
    coord = r.copy()
    
    # Subtract whole multiples of L using integer division and subtraction
    multiples_of_L = np.floor_divide(r, L)
    coord -= multiples_of_L * L
    
    # Adjust negative coordinates to ensure they are within the [0, L) range
    coord = np.where(coord < 0, coord + L, coord)
    
    return coord


try:
    targets = process_hdf5_to_tuple('60.1', 3)
    target = targets[0]
    particle_position = np.array([10.5, -1.2, 20.3])
    box_length = 10.0
    # Applying the wrap function
    assert np.allclose(wrap(particle_position, box_length), target) # Expected output: [0.5, 8.8, 0.3]

    target = targets[1]
    particle_position1 = np.array([10.0, 5.5, -0.1])
    box_length1 = 10.0
    # Applying the wrap function
    assert np.allclose(wrap(particle_position1, box_length1), target)  # Expected output: [0.0, 5.5, 9.9]

    target = targets[2]
    particle_position2 = np.array([23.7, -22.1, 14.3])
    box_length2 = 10.0
    # Applying the wrap function
    assert np.allclose(wrap(particle_position2, box_length2), target)  # Expected output: [3.7, 7.9, 4.3]

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e