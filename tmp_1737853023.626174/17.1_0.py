import sympy as sp
import numpy as np



# Background: In computational physics and numerical simulations, especially in the context of density of states calculations,
# it is often necessary to work with energy differences between various points in a system. In this problem, we are dealing
# with a tetrahedron in energy space, where each vertex has a specific energy value. The task is to compute the differences
# between these energy values and a given iso-value energy level, E. These differences are crucial for further calculations
# such as interpolation or integration over the energy space. The sympy library is used to symbolically represent these
# differences, allowing for symbolic manipulation and evaluation later in the process.

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
    
    # Create a dictionary to store sympy symbols
    symbols = {}
    
    # Create a dictionary to map symbols to their actual values
    value_map = {}
    
    # Define the energy levels including the iso-value energy
    energies = [energy] + energy_vertices
    
    # Populate the array with sympy symbols and calculate the differences
    for i in range(5):
        for j in range(5):
            if i != j:
                # Create a sympy symbol for the energy difference
                symbol_name = f'e_{j}{i}'
                symbol = sp.symbols(symbol_name)
                symbols[symbol_name] = symbol
                
                # Calculate the energy difference
                e_ji_array[j, i] = symbol
                value_map[symbol] = energies[j] - energies[i]
            else:
                # Diagonal elements are zero since e_ii = 0
                e_ji_array[j, i] = 0
    
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
