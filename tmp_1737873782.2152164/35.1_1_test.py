from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools



def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    
    # Constants
    electron_mass = 9.109e-31  # kg
    planck_constant = 6.626e-34  # J*s
    speed_of_light = 3e8  # m/s
    
    # Convert L from nm to meters
    L_meters = L * 1e-9
    
    # Effective mass
    effective_mass = mr * electron_mass
    
    # Calculate ground state energy in joules using the formula:
    # E_n = (n^2 * π^2 * h^2) / (2 * m * L^2) for n=1
    ground_state_energy = (planck_constant**2 * np.pi**2) / (2 * effective_mass * L_meters**2)
    
    # Convert energy to wavelength using the formula:
    # E = hc / λ => λ = hc / E
    wavelength_meters = (planck_constant * speed_of_light) / ground_state_energy
    
    # Convert wavelength from meters to nanometers
    lmbd = wavelength_meters * 1e9
    
    return lmbd


try:
    targets = process_hdf5_to_tuple('35.1', 3)
    target = targets[0]
    assert np.allclose(ground_state_wavelength(5,0.6), target)

    target = targets[1]
    assert np.allclose(ground_state_wavelength(10,0.6), target)

    target = targets[2]
    assert np.allclose(ground_state_wavelength(10,0.06), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e