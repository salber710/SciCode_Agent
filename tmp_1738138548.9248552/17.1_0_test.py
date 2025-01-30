from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import sympy as sp
import numpy as np



# Background: In computational physics and numerical simulations, especially in the context of density of states calculations,
# it is often necessary to work with energy differences between various points in a system. In this problem, we are dealing
# with a tetrahedron in energy space, where each vertex of the tetrahedron has a specific energy value. The task is to compute
# the differences between these energy values and a given iso-value energy level, E. These differences are crucial for
# understanding how the energy levels vary across the tetrahedron and are used in further calculations such as interpolation
# or integration over the energy space. The sympy library is used to symbolically represent these differences, which allows
# for symbolic manipulation and evaluation later in the process.



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
    
    # Initialize a 5x5 array for energy differences
    e_ji_array = np.zeros((5, 5), dtype=object)
    
    # Create a list of energies including the iso-value energy
    energies = [energy] + energy_vertices
    
    # Initialize dictionaries for symbols and value mapping
    symbols = {}
    value_map = {}
    
    # Populate the array and dictionaries
    for i in range(5):
        for j in range(5):
            if i != j:
                # Calculate the energy difference
                e_ji = energies[j] - energies[i]
                
                # Create a sympy symbol for this difference
                symbol_name = f'e_{j}{i}'
                symbol = sp.symbols(symbol_name)
                
                # Store the symbol in the array
                e_ji_array[j, i] = symbol
                
                # Map the symbol name to the symbol
                symbols[symbol_name] = symbol
                
                # Map the symbol to its actual value
                value_map[symbol] = e_ji
    
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