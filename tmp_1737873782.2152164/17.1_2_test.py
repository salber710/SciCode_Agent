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
    
    # Initialize a 5x5 array to store the differences
    e_ji_array = np.zeros((5, 5))
    
    # Initialize dictionaries for symbols and their values
    symbols = {}
    value_map = {}

    # The first row and column are differences with respect to the energy level E
    energies = [energy] + energy_vertices
    
    # Iterate over the indices (i, j) to fill in the array and create sympy symbols
    for i in range(5):
        for j in range(5):
            if j > i:  # Only fill for j > i to avoid redundant calculations
                e_ji_array[j, i] = energies[j] - energies[i]
                
                # Create a sympy symbol for this difference
                symbol_name = f"e_{j}{i}"
                symbol = sp.symbols(symbol_name)
                
                symbols[symbol_name] = symbol
                value_map[symbol] = e_ji_array[j, i]

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