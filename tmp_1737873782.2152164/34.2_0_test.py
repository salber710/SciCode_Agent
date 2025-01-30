from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def Fermi(N_a, N_d, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    
    # Thermal voltage at room temperature (given)
    V_t = 0.0259
    
    # Built-in potential (V_bi) using the doping concentrations
    V_bi = V_t * np.log(N_a * N_d / n_i**2)
    
    # Compute the Fermi levels in terms of the intrinsic level E_i
    # phi_p = V_t * ln(N_a / n_i) for p-type
    phi_p = V_t * np.log(N_a / n_i)
    
    # phi_n = V_t * ln(N_d / n_i) for n-type
    phi_n = V_t * np.log(N_d / n_i)
    
    return phi_p, phi_n



def depletion(N_a, N_d, n_i, e_r):
    '''This function calculates the depletion width in both n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    '''
    
    # Constants
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19  # Electron charge in C
    
    # Compute the built-in potential using the function Fermi
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    V_bi = phi_n - phi_p  # Built-in potential is the difference of Fermi levels
    
    # Total permittivity
    epsilon_s = e_r * epsilon_0
    
    # Depletion widths using Poisson's equation
    xp = np.sqrt(2 * epsilon_s * V_bi * N_d / (q * N_a * (N_a + N_d)))
    xn = np.sqrt(2 * epsilon_s * V_bi * N_a / (q * N_d * (N_a + N_d)))
    
    return xn, xp


try:
    targets = process_hdf5_to_tuple('34.2', 3)
    target = targets[0]
    assert np.allclose(depletion(2*10**17,3*10**17,10**12,15), target)

    target = targets[1]
    assert np.allclose(depletion(1*10**17,2*10**17,10**12,15), target)

    target = targets[2]
    assert np.allclose(depletion(2*10**17,3*10**17,2*10**11,15), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e