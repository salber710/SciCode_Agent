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



# Background: In computational physics, the density of states (DOS) is a critical concept used to describe the number of states
# available to a system at each energy level. When dealing with a tetrahedron in energy space, the linear tetrahedron method
# is often used to perform DOS integration. This method involves calculating the contribution of each tetrahedron to the DOS
# by considering the energy values at its vertices and the iso-value energy level. The integration process requires evaluating
# the volume of the region within the tetrahedron where the energy is less than or equal to the iso-value energy. Depending on
# the relative magnitudes of the iso-value energy and the vertex energies, different cases arise, each requiring specific
# calculations to determine the contribution to the DOS.

def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''



    # Sort the energy vertices to ensure the order is correct
    energy_vertices = sorted(energy_vertices)
    
    # Initialize the result of the integration
    result = 0.0
    
    # Calculate the energy differences
    e0 = energy
    e1, e2, e3, e4 = energy_vertices
    e_ji = [e0 - e1, e0 - e2, e0 - e3, e0 - e4]
    
    # Determine the case based on the energy level
    if e0 <= e1:
        # Case 1: E is below all vertex energies, no contribution
        result = 0.0
    elif e0 <= e2:
        # Case 2: E is between e1 and e2
        result = (e_ji[0]**3) / (6 * (e2 - e1) * (e3 - e1) * (e4 - e1))
    elif e0 <= e3:
        # Case 3: E is between e2 and e3
        result = ((e_ji[0]**3) / (6 * (e2 - e1) * (e3 - e1) * (e4 - e1)) +
                  (e_ji[1]**3) / (6 * (e3 - e2) * (e4 - e2) * (e3 - e1)))
    elif e0 <= e4:
        # Case 4: E is between e3 and e4
        result = ((e_ji[0]**3) / (6 * (e2 - e1) * (e3 - e1) * (e4 - e1)) +
                  (e_ji[1]**3) / (6 * (e3 - e2) * (e4 - e2) * (e3 - e1)) +
                  (e_ji[2]**3) / (6 * (e4 - e3) * (e4 - e2) * (e4 - e1)))
    else:
        # Case 5: E is above all vertex energies
        result = 1.0 / 6.0  # Full contribution of the tetrahedron

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
