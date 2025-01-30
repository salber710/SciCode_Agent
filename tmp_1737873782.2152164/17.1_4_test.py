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

    # Initialize a 5x5 array for e_ji
    eji_array = np.zeros((5, 5), dtype=object)
    
    # Combine energy and energy vertices into a single list
    energies = [energy] + energy_vertices
    
    # Initialize dictionaries for symbols and their corresponding values
    symbols = {}
    value_map = {}
    
    # Populate the e_ji array and create sympy symbols
    for i in range(5):
        for j in range(5):
            if i != j:
                # Calculate the energy difference
                eji = energies[j] - energies[i]
                
                # Create a symbol for e_ji
                symbol_name = f'e_{j}{i}'
                symbol = sp.symbols(symbol_name)
                
                # Store the symbol and its calculated value
                symbols[symbol_name] = symbol
                value_map[symbol] = eji
                
                # Update the eji_array with the symbol
                eji_array[j, i] = symbol
    
    return symbols, value_map


try:
    targets = process_hdf5_to_tuple('17.1', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    assert cmp_tuple_or_list(init_eji_array(10,[4,6,8,10]), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    assert cmp_tuple_or_list(init_eji_array(1,[1,2,3,4]), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    assert cmp_tuple_or_list(init_eji_array(2.2,[1.2,2.2,3.4,5.5]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e