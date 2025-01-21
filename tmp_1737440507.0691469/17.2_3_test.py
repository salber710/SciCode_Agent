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



# Background: The density of states (DOS) is a concept used in physics and chemistry to describe the number of states per interval of energy at each energy level available to be occupied. In the context of a tetrahedral mesh, the DOS integration can be performed using the linear tetrahedron method. This method involves evaluating the contributions of energy states within each tetrahedron by considering the energy at the iso-value surface compared to the energies at the vertices. Depending on the relative values of the iso-value energy E and vertex energies εᵢ, the contribution to the DOS can vary. The integration considers these scenarios to compute the total density of states accurately.

def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''

    # Sort the vertices' energies
    energy_vertices = sorted(energy_vertices)
    
    # Initialize DOS result
    result = 0.0

    # Unpack sorted energy vertices
    epsilon_1, epsilon_2, epsilon_3, epsilon_4 = energy_vertices

    # Case 1: E < ε₁
    if energy < epsilon_1:
        result = 0.0
    
    # Case 2: ε₁ <= E < ε₂
    elif epsilon_1 <= energy < epsilon_2:
        result = (energy - epsilon_1)**3 / ((epsilon_4 - epsilon_1) * (epsilon_3 - epsilon_1) * (epsilon_2 - epsilon_1))
    
    # Case 3: ε₂ <= E < ε₃
    elif epsilon_2 <= energy < epsilon_3:
        term1 = (epsilon_2 - epsilon_1) + (energy - epsilon_2)
        term2 = (epsilon_3 - epsilon_1) + (energy - epsilon_3)
        result = ((term1 * term2) / ((epsilon_4 - epsilon_1) * (epsilon_3 - epsilon_1) * (epsilon_3 - epsilon_2))) * (epsilon_3 - energy)
    
    # Case 4: ε₃ <= E < ε₄
    elif epsilon_3 <= energy < epsilon_4:
        result = 1.0 - (epsilon_4 - energy)**3 / ((epsilon_4 - epsilon_1) * (epsilon_4 - epsilon_2) * (epsilon_4 - epsilon_3))
    
    # Case 5: E >= ε₄
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
