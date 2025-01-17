import sympy as sp
import numpy as np



# Background: In computational physics and numerical simulations, especially in the context of density of states calculations,
# it is often necessary to work with energy differences between various points in a system. In this problem, we are dealing
# with a tetrahedron in energy space, where each vertex has a specific energy value. The task is to compute the differences
# between these energies and an iso-value energy level, and represent these differences using symbolic computation for further
# analysis. The energy differences are denoted as ε_ji = ε_j - ε_i, where ε_0 is the iso-value energy and ε_1, ε_2, ε_3, ε_4
# are the energies at the vertices of the tetrahedron. We will use the sympy library to create symbolic representations of
# these energy differences, which will allow for symbolic manipulation and evaluation in later steps.



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
    
    # Dictionary to store sympy symbols
    symbols = {}
    
    # Dictionary to map symbols to their actual values
    value_map = {}
    
    # Populate the array and dictionaries
    for i in range(5):
        for j in range(5):
            if i != j:
                # Create a symbol for the energy difference ε_ji
                symbol_name = f'e_{j}{i}'
                symbol = sp.symbols(symbol_name)
                
                # Calculate the energy difference
                e_ji_value = energies[j] - energies[i]
                
                # Store the symbol in the array
                e_ji_array[j, i] = symbol
                
                # Map the symbol name to the symbol
                symbols[symbol_name] = symbol
                
                # Map the symbol to its actual value
                value_map[symbol] = e_ji_value
    
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
