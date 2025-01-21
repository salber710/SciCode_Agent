import sympy as sp
import numpy as np



# Background: In the context of computational physics or chemistry, we often deal with energy values at different points in a system. 
# In this case, we are working with a tetrahedron, which has four vertices, each with an associated energy value. 
# The energy on an iso-value surface is given as ε₀ = E, and each vertex of the tetrahedron has an energy εᵢ. 
# The task involves calculating the energy differences ε_{ji} = εⱼ - εᵢ for i, j = 0,...,4. These differences are useful for 
# analyzing variations in energy across the tetrahedron and are often used in numerical integration methods, such as those 
# employed in density of states calculations. We will be leveraging the sympy library to handle symbolic representation and manipulation 
# of these differences, which is crucial for later evaluation and integration processes.

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


    
    # Create a 5x5 numpy array for ε_{ji}
    epsilon_differences = np.zeros((5, 5), dtype=object)
    
    # Create sympy symbols and a value mapping dictionary
    symbols = {}
    value_map = {}
    
    # List of energies including the iso-value surface energy ε₀ = E
    energies = [energy] + energy_vertices
    
    # Populate the ε_{ji} array and create sympy symbols and value mappings
    for i in range(5):
        for j in range(5):
            # Calculate ε_{ji} = εⱼ - εᵢ
            epsilon_differences[j][i] = energies[j] - energies[i]
            
            # Create symbolic representation
            symbol_name = f"e_{j}{i}"
            symbol = sp.symbols(symbol_name)
            symbols[symbol_name] = symbol
            value_map[symbol] = epsilon_differences[j][i]
    
    return symbols, value_map

from scicode.parse.parse import process_hdf5_to_tuple
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
