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



# Background: The density of states (DOS) is a function that describes the number of states that are available to be occupied at each energy level within a system. 
# In the linear tetrahedron method, DOS integration is carried out over each tetrahedron in a discretized model of the system's energy space. 
# The energies at the vertices of the tetrahedron are compared to the iso-value surface energy E. The integration involves calculating the contribution 
# to the DOS from the tetrahedron based on how E compares to the vertex energies. Different cases arise depending on whether E is below all vertex energies, 
# above all, or between some of them. The integration takes into account these cases and uses the volume of the tetrahedron and the linear interpolation of energies 
# to compute the integral contribution to the DOS.

def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''



    # Sort the energy vertices to simplify the comparison logic
    energy_vertices = sorted(energy_vertices)
    
    # Calculate volume of the tetrahedron in energy space for normalization
    # Assuming a unit volume for simplicity, as we are interested in the relative contribution
    V_tetra = 1.0

    # Calculate energy differences
    epsilon_differences = np.zeros(4)
    for i in range(4):
        epsilon_differences[i] = energy_vertices[i] - energy
    
    # Determine position of E relative to vertex energies
    if energy < energy_vertices[0]:
        # Case 1: E is below all vertex energies
        result = 0.0
    elif energy >= energy_vertices[3]:
        # Case 5: E is above all vertex energies
        result = V_tetra
    else:
        # Intermediate cases: E is between some of the vertex energies
        # Calculate partial volumes where E contributes
        result = 0.0
        for i in range(3):
            if energy_vertices[i] <= energy < energy_vertices[i+1]:
                # Linear interpolation within the sub-tetrahedron
                # Contribution from the part of the tetrahedron where energy is within [v_i, v_{i+1}]
                # Fractional volume between i-th and (i+1)-th vertices
                frac = (energy - energy_vertices[i]) / (energy_vertices[i+1] - energy_vertices[i])
                # Volume contribution using linear interpolation factor
                result += frac * (V_tetra / 3.0)
    
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
