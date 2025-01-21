import sympy as sp
import numpy as np



# Background: In computational physics, especially in methods involving numerical integration over 
# discretized domains (like tetrahedra in 3D space), it is often useful to compute differences between 
# energy levels. These differences help in various calculations, such as determining where an iso-value 
# surface (a surface of constant energy) intersects a tetrahedron. In this problem, we are tasked with 
# initializing a 5x5 matrix of energy differences for a given energy level and the energy at vertices 
# of a tetrahedron. The matrix will include differences between each vertex energy and the iso-value 
# surface energy, as well as between each pair of vertex energies. We will use SymPy to create symbolic 
# representations of these differences to facilitate further symbolic manipulation or evaluation.



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
    
    # Initialize a 5x5 array to store energy differences e_ji
    e_ji_array = np.zeros((5, 5), dtype=object)

    # Create a dictionary to hold the sympy symbols
    symbols = {}
    
    # Create a dictionary to map sympy symbols to their actual values
    value_map = {}
    
    # Define the energies for convenience
    energies = [energy] + energy_vertices
    
    # Populate the array with symbolic variables and their values
    for i in range(5):
        for j in range(5):
            if i != j:
                # Define a sympy symbol for the difference
                symbol_name = f'e_{j}{i}'
                symbol = sp.Symbol(symbol_name)
                
                # Calculate the energy difference
                energy_difference = energies[j] - energies[i]
                
                # Store the symbol in the array
                e_ji_array[j, i] = symbol
                
                # Add the symbol and its value to the dictionaries
                symbols[symbol_name] = symbol
                value_map[symbol] = energy_difference
    
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
