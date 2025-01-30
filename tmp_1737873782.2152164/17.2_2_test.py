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
    # Sort the energy vertices
    energy_vertices = sorted(energy_vertices)
    
    # Unpack the sorted vertices
    e1, e2, e3, e4 = energy_vertices
    
    # Calculate the energy differences
    e_ji_map = {
        'e_10': energy - e1,
        'e_20': energy - e2,
        'e_30': energy - e3,
        'e_40': energy - e4,
        'e_21': e2 - e1,
        'e_31': e3 - e1,
        'e_41': e4 - e1,
        'e_32': e3 - e2,
        'e_42': e4 - e2,
        'e_43': e4 - e3,
    }

    # Check the conditions for integration based on energy comparisons
    if energy < e1 or energy > e4:
        return 0.0  # No contribution if energy is outside the range of vertex energies
    
    # Calculate the norm of the gradient
    grad_norm = np.sqrt(e_ji_map['e_21']**2 + e_ji_map['e_31']**2 + e_ji_map['e_41']**2)
    
    # Calculate the volume of the tetrahedron in a normalized space
    Omega_T = abs(e_ji_map['e_21'] * e_ji_map['e_32'] * e_ji_map['e_43']) / 6.0
    
    # Calculate the contribution to the DOS
    if e1 <= energy < e2:
        result = (6 * Omega_T / grad_norm) * (e_ji_map['e_10'] / e_ji_map['e_21'])
    elif e2 <= energy < e3:
        result = (6 * Omega_T / grad_norm) * (1 + (e_ji_map['e_20'] / e_ji_map['e_32']))
    elif e3 <= energy < e4:
        result = (6 * Omega_T / grad_norm) * (2 + (e_ji_map['e_30'] / e_ji_map['e_43']))
    else:
        result = 0.0
    
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