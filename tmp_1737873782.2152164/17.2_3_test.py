from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import sympy as sp
import numpy as np



def init_eji_array(energy, energy_vertices):
    '''Initialize and populate a 5x5 array for storing e_ji variables, and map e_ji values
    to sympy symbols for later evaluation.
    Inputs:
    - energy: A float representing the energy level for density of states integration.
    - energy_vertices: A list of floats representing energy values at tetrahedron vertices.
    Outputs:
    - symbols: A dictionary mapping sympy symbol names to symbols.
    - value_map: A dictionary mapping symbols to their actual values (float).
    '''

    # Total number of vertices including the iso-value energy level
    num_vertices = len(energy_vertices) + 1  # 4 vertices + 1 iso-value energy level
    energies = [energy] + energy_vertices  # Include the iso-value energy level

    # Initialize a 5x5 array for storing energy differences
    e_ji_array = np.zeros((num_vertices, num_vertices), dtype=float)

    # Create dictionaries to hold the sympy symbols and their corresponding values
    symbols = {}
    value_map = {}

    # Populate the array with the energy differences and create sympy symbols
    for i in range(num_vertices):
        for j in range(num_vertices):
            e_ji = energies[j] - energies[i]
            e_ji_array[j, i] = e_ji
            # Create a sympy symbol for this difference
            symbol_name = f"e_{j}{i}"
            symbol = sp.symbols(symbol_name)
            symbols[symbol_name] = symbol
            value_map[symbol] = e_ji

    return symbols, value_map



def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''
    # Sort the energy vertices to ensure the order is correct
    energy_vertices.sort()
    
    # Unpack the sorted energy vertices
    epsilon1, epsilon2, epsilon3, epsilon4 = energy_vertices

    # Initialize the result of the integration
    result = 0.0

    # Check the position of energy E relative to the vertices
    if energy <= epsilon1 or energy >= epsilon4:
        # If E is outside the range of the tetrahedron vertices, contribution is zero
        return result

    # Calculate the volume of the tetrahedron
    # For simplicity, assume a unit volume, as we'll normalize later
    Omega_T = 1.0 

    # Calculate |\nabla \varepsilon(e, u, v)|
    grad_norm = np.sqrt((epsilon2 - epsilon1)**2 + (epsilon3 - epsilon1)**2 + (epsilon4 - epsilon1)**2)

    # Contribution to DOS from each region based on the position of E
    if epsilon1 < energy < epsilon2:
        # Region 1: E is between epsilon1 and epsilon2
        result = (6 * Omega_T / grad_norm) * ((energy - epsilon1) / (epsilon2 - epsilon1)**2)
    elif epsilon2 <= energy < epsilon3:
        # Region 2: E is between epsilon2 and epsilon3
        result = (6 * Omega_T / grad_norm) * (1 / (epsilon3 - epsilon2))
    elif epsilon3 <= energy < epsilon4:
        # Region 3: E is between epsilon3 and epsilon4
        result = (6 * Omega_T / grad_norm) * ((epsilon4 - energy) / (epsilon4 - epsilon3)**2)

    # Normalize the result by the volume of the BZ
    Omega_BZ = 1.0  # Assuming unit Brillouin Zone volume for simplicity
    result /= Omega_BZ

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