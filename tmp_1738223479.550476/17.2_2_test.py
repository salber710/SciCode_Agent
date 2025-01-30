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

    # Sort the vertices by energy values and get the sorted indices
    sorted_indices = sorted(range(len(energy_vertices)), key=lambda i: energy_vertices[i])
    sorted_energies = [energy_vertices[i] for i in sorted_indices]
    
    # Initialize the integrated DOS result
    result = 0.0
    
    # Check cases for integration based on energy level
    if energy < sorted_energies[0]:
        return result  # Energy is below the lowest vertex, so contribution is zero
    
    if energy >= sorted_energies[3]:
        return 1.0  # Energy is above the highest vertex, full contribution
    
    # For each energy interval, determine contribution
    for i in range(3):
        e_low = sorted_energies[i]
        e_high = sorted_energies[i + 1]
        
        if e_low <= energy < e_high:
            # Calculate the contribution using linear interpolation
            e_range = e_high - e_low
            if e_range != 0:
                fraction = (energy - e_low) / e_range
                contribution = (fraction**2) * (3 - 2*fraction)
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