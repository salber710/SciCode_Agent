from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import sympy as sp
import numpy as np


def init_eji_array(energy, energy_vertices):
    energies = [energy] + energy_vertices
    symbols = {}
    value_map = {}
    
    for j in range(5):
        for i in range(5):
            if i != j:
                symbol = sp.Symbol(f'ΔE{j}{i}')
                symbols[f'ΔE{j}{i}'] = symbol
                value_map[symbol] = energies[j] - energies[i]
    
    return symbols, value_map



# Background: 
# The Density of States (DOS) integration within a tetrahedron is a crucial step in the linear tetrahedron method, 
# which is used in electronic structure calculations to determine the DOS at a given energy level. 
# The method involves integrating over the volume of a tetrahedron in k-space, where the energy varies linearly 
# between the vertices. The integration is performed by considering the position of the iso-energy surface 
# relative to the energies at the vertices of the tetrahedron. Depending on the energy value E relative to 
# the vertex energies, different cases arise:
# 1. E < ε1: The iso-energy surface does not intersect the tetrahedron.
# 2. ε1 <= E < ε2: The iso-energy surface intersects the tetrahedron, forming a smaller tetrahedron.
# 3. ε2 <= E < ε3: The iso-energy surface intersects the tetrahedron, forming a truncated tetrahedron.
# 4. ε3 <= E < ε4: The iso-energy surface intersects the tetrahedron, forming a different truncated tetrahedron.
# 5. E >= ε4: The iso-energy surface fully encompasses the tetrahedron.
# The integration involves calculating the volume of these intersected regions and using them to compute the DOS.

def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''
    # Sort the vertex energies
    energy_vertices = sorted(energy_vertices)
    ε1, ε2, ε3, ε4 = energy_vertices

    # Initialize the result
    result = 0.0

    # Case 1: E < ε1
    if energy < ε1:
        result = 0.0

    # Case 2: ε1 <= E < ε2
    elif ε1 <= energy < ε2:
        # Calculate the volume of the smaller tetrahedron
        result = (energy - ε1)**3 / ((ε2 - ε1) * (ε3 - ε1) * (ε4 - ε1))

    # Case 3: ε2 <= E < ε3
    elif ε2 <= energy < ε3:
        # Calculate the volume of the truncated tetrahedron
        result = ((ε2 - ε1) + 3 * (energy - ε2)) * (energy - ε1)**2 / (3 * (ε3 - ε1) * (ε4 - ε1))

    # Case 4: ε3 <= E < ε4
    elif ε3 <= energy < ε4:
        # Calculate the volume of the different truncated tetrahedron
        result = ((ε3 - ε1) + 3 * (energy - ε3)) * (energy - ε1)**2 / (3 * (ε4 - ε1) * (ε3 - ε1))

    # Case 5: E >= ε4
    else:
        # The iso-energy surface fully encompasses the tetrahedron
        result = 1.0

    return result


try:
    targets = process_hdf5_to_tuple('17.2', 5)
    target = targets[0]
    energy = 1.5
    energy_vertices = [1, 2, 3, 4] #e1-e4
    assert np.allclose(float(integrate_DOS(energy, energy_vertices)), target)

    target = targets[1]
    energy = 2.7
    energy_vertices = [1, 2, 3, 4] #e1-e4
    assert np.allclose(float(integrate_DOS(energy, energy_vertices)), target)

    target = targets[2]
    energy = 3.6
    energy_vertices = [1, 2, 3, 4] #e1-e4
    assert np.allclose(float(integrate_DOS(energy, energy_vertices)), target)

    target = targets[3]
    energy = 5
    energy_vertices = [1, 2, 3, 4] #e1-e4
    assert np.allclose(float(integrate_DOS(energy, energy_vertices)), target)

    target = targets[4]
    energy = 0.9
    energy_vertices = [1, 2, 3, 4] #e1-e4
    assert (float(integrate_DOS(energy, energy_vertices)) == 0) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e