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


    # Energy values including iso-surface energy as the first element
    energies = [energy] + energy_vertices

    # Initialize a dictionary for symbols and a separate one for values
    symbols = {}
    value_map = {}

    # Use enumeration for better readability and to access indices directly
    for j, energy_j in enumerate(energies):
        for i, energy_i in enumerate(energies):
            # Construct the symbol name
            symbol_name = f'e_{j}{i}'

            # Create the sympy symbol
            symbol = sp.symbols(symbol_name)

            # Calculate the energy difference
            e_ji = energy_j - energy_i

            # Store the symbol in the dictionary
            symbols[symbol_name] = symbol

            # Map the symbol to its computed value
            value_map[symbol] = e_ji

    return symbols, value_map



def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''

    # Sort the energy vertices and keep track of their indices
    indexed_vertices = sorted(enumerate(energy_vertices), key=lambda x: x[1])
    sorted_indices, sorted_energies = zip(*indexed_vertices)

    # Initialize the result
    result = 0.0

    # Case 1: Energy is below the lowest vertex energy
    if energy < sorted_energies[0]:
        return result

    # Case 2: Energy is above the highest vertex energy
    if energy >= sorted_energies[3]:
        return 1.0

    # Case 3: Energy is between vertex energies
    # Calculate contributions based on the position of the energy
    for i in range(3):
        e_lower = sorted_energies[i]
        e_upper = sorted_energies[i + 1]

        if e_lower <= energy < e_upper:
            # Use a cubic formula for variety
            delta_e = e_upper - e_lower
            if delta_e != 0:
                factor = (energy - e_lower) / delta_e
                contribution = 3 * (factor**2) - 2 * (factor**3)  # f(x) = 3x^2 - 2x^3
                result += contribution

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