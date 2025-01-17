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



# Background: In computational physics, the density of states (DOS) is a critical concept used to describe the number of 
# states available to a system at each energy level. When dealing with a tetrahedron in energy space, the linear tetrahedron 
# method is often used to integrate the DOS over the volume of the tetrahedron. This method involves evaluating the DOS at 
# an iso-energy surface, which is a surface of constant energy, and comparing it to the energies at the vertices of the 
# tetrahedron. The integration process requires considering different cases based on the relative position of the iso-energy 
# level with respect to the vertex energies. The goal is to compute the contribution of the tetrahedron to the DOS at a 
# given energy level by integrating over the tetrahedron's volume.

def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''

    
    # Sort the energy vertices to ensure ε_1 < ε_2 < ε_3 < ε_4
    energy_vertices = sorted(energy_vertices)
    
    # Unpack the sorted energies
    e1, e2, e3, e4 = energy_vertices
    
    # Initialize the result of the integration
    result = 0.0
    
    # Case 1: E < ε_1
    if energy < e1:
        result = 0.0
    
    # Case 2: ε_1 <= E < ε_2
    elif e1 <= energy < e2:
        result = ((energy - e1)**3) / ((e4 - e1) * (e3 - e1) * (e2 - e1))
    
    # Case 3: ε_2 <= E < ε_3
    elif e2 <= energy < e3:
        result = ((e2 - e1)**3 + 3 * (e2 - e1)**2 * (energy - e2) + 3 * (e2 - e1) * (energy - e2)**2 + (energy - e2)**3) / ((e4 - e1) * (e3 - e1) * (e3 - e2))
    
    # Case 4: ε_3 <= E < ε_4
    elif e3 <= energy < e4:
        result = ((e2 - e1)**3 + 3 * (e2 - e1)**2 * (e3 - e2) + 3 * (e2 - e1) * (e3 - e2)**2 + (e3 - e2)**3 + 3 * (e3 - e2)**2 * (energy - e3) + 3 * (e3 - e2) * (energy - e3)**2 + (energy - e3)**3) / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    
    # Case 5: E >= ε_4
    else:
        result = 1.0
    
    return result


from scicode.parse.parse import process_hdf5_to_tuple

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
