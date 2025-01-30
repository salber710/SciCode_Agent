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

    # Sort the energy vertices and their indices
    sorted_indices = sorted(range(len(energy_vertices)), key=lambda i: energy_vertices[i])
    sorted_energies = [energy_vertices[i] for i in sorted_indices]

    # Initialize result
    result = 0.0

    # Define integration weights based on energy position
    def compute_weights(e, e1, e2, e3, e4):
        if e < e1:
            return [0, 0, 0, 0]
        elif e >= e4:
            return [1, 1, 1, 1]
        else:
            d1 = e2 - e1
            d2 = e3 - e2
            d3 = e4 - e3
            if e1 <= e < e2:
                w1 = ((e - e1) ** 2) / (d1 ** 2)
                w2 = (e - e1) / d1
                return [w1, w2, 0, 0]
            elif e2 <= e < e3:
                w2 = ((e - e2) ** 2) / (d2 ** 2)
                w3 = (e - e2) / d2
                return [1, w2, w3, 0]
            elif e3 <= e < e4:
                w3 = ((e - e3) ** 2) / (d3 ** 2)
                w4 = (e - e3) / d3
                return [1, 1, w3, w4]
    
    # Calculate weights for the sorted energies and integrate
    weights = compute_weights(energy, *sorted_energies)
    result = sum(weights) / len(weights)

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