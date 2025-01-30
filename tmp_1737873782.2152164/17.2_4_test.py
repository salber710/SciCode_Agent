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

    # Total number of vertices including the iso-value energy level
    num_vertices = len(energy_vertices) + 1  # 4 vertices + 1 iso-value energy level
    energies = [energy] + energy_vertices  # Include the iso-value energy level

    # Initialize a 5x5 array for storing energy differences
    e_ji_array = np.zeros((num_vertices, num_vertices), dtype=float)

    # Create dictionaries to hold the sympy symbols and their corresponding values
    symbols = {}
    value_map = {}

    # Populate the array with the energy differences and create sympy symbols
    for i in range(num_vertices):
        for j in range(num_vertices):
            e_ji = energies[j] - energies[i]
            e_ji_array[j, i] = e_ji
            # Create a sympy symbol for this difference
            symbol_name = f"e_{j}{i}"
            symbol = sp.symbols(symbol_name)
            symbols[symbol_name] = symbol
            value_map[symbol] = e_ji

    return symbols, value_map



def integrate_DOS(energy, energy_vertices):
    '''Input:
    energy: a float number representing the energy value at which the density of states will be integrated
    energy_vertices: a list of float numbers representing the energy values at the four vertices of a tetrahedron when implementing the linear tetrahedron method
    Output:
    result: a float number representing the integration results of the density of states
    '''
    # Assumptions: energy_vertices are sorted in increasing order
    assert len(energy_vertices) == 4, "There must be exactly 4 energy vertices."

    # Initialize and get sympy symbols and value map using the provided function
    symbols, value_map = init_eji_array(energy, energy_vertices)

    # Extract energy differences needed for the integration
    e_21 = value_map[symbols['e_21']]
    e_31 = value_map[symbols['e_31']]
    e_41 = value_map[symbols['e_41']]

    # Calculate the norm of the gradient using the derived formula
    grad_norm = np.sqrt(e_21**2 + e_31**2 + e_41**2)

    # Calculate the volume of the tetrahedron (this is a placeholder, replace with actual volume computation)
    # The actual computation of Omega_T should be done based on the coordinates of the tetrahedron vertices.
    Omega_T = 1  # Placeholder for the tetrahedron volume

    # Calculate the contribution to the DOS from this tetrahedron
    if energy < energy_vertices[0] or energy > energy_vertices[3]:
        # If the energy is outside the range of vertex energies, contribution is zero
        result = 0.0
    else:
        # Calculate the contribution to the DOS
        # This is based on the formula provided for DOS integration within a single tetrahedron
        Omega_BZ = 1  # Placeholder for the volume of the Brillouin zone
        result = (6 * Omega_T / Omega_BZ) * (1 / grad_norm)

    return result


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e