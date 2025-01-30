from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def gain(nw, Gamma_w, alpha, L, R1, R2):
    '''Calculates the single peak gain coefficient g_w.
    Input:
    nw (float): Quantum well number.
    Gamma_w (float): Confinement factor of the waveguide.
    alpha (float): Internal loss coefficient in cm^-1.
    L (float): Cavity length in cm.
    R1 (float): The reflectivities of mirror 1.
    R2 (float): The reflectivities of mirror 2.
    Output:
    gw (float): Gain coefficient g_w.
    '''


    # Calculate the reflectivity product
    reflectivity_product = R1 * R2

    # Use a different approach with exponential and logarithmic transformations
    exponential_reflectivity = exp(log(reflectivity_product) / 2)
    inverse_exponential_reflectivity = 1 / exponential_reflectivity

    # Calculate the mirror loss using the transformed reflectivity
    mirror_loss = (1 / L) * log(inverse_exponential_reflectivity)

    # Calculate the threshold gain coefficient G_th
    gw = (alpha + mirror_loss) / (Gamma_w * nw)

    return gw


def current_density(gw, g0, J0):
    '''Calculates the current density J_w as a function of the gain coefficient g_w.
    Input:
    gw (float): Gain coefficient.
    g0 (float): Empirical gain coefficient.
    J0 (float): Empirical factor in the current density equation.
    Output:
    Jw (float): Current density J_w.
    '''
    # Implement a sine-based relation for the current density

    # Ensure gw and g0 are positive to avoid math domain errors
    if gw <= 0 or g0 <= 0:
        raise ValueError("Gain coefficients must be positive.")
        
    # Use a sine function to model the current density
    Jw = J0 * (math.sin(math.pi * gw / (2 * g0)) + 1)
    
    return Jw



def threshold_current(nw, Gamma_w, alpha, L, R1, R2, g0, J0, eta, w):
    '''Calculates the threshold current.
    Input:
    nw (float): Quantum well number.
    Gamma_w (float): Confinement factor of the waveguide.
    alpha (float): Internal loss coefficient.
    L (float): Cavity length.
    R1 (float): The reflectivities of mirror 1.
    R2 (float): The reflectivities of mirror 2.
    g0 (float): Empirical gain coefficient.
    J0 (float): Empirical factor in the current density equation.
    eta (float): injection quantum efficiency.
    w (float): device width.
    Output:
    Ith (float): threshold current Ith
    '''

    
    # Calculate the reflectivity loss in a unique manner
    reflectivity_loss = (np.log1p(R1) + np.log1p(R2)) / (2 * L)
    
    # Determine the total necessary gain using a different coefficient combination
    necessary_gain = (alpha + reflectivity_loss) * Gamma_w
    
    # Calculate the gain coefficient from the gain function
    gain_coefficient = gain(nw, Gamma_w, alpha, L, R1, R2)
    
    # Adjust the gain coefficient to meet the necessary gain
    gain_coefficient = gain_coefficient if gain_coefficient >= necessary_gain else necessary_gain
    
    # Compute the injected current density using the current_density function
    injected_current_density = current_density(gain_coefficient, g0, J0)
    
    # Calculate a different effective area considering squared efficiency
    effective_area = (L * w * (1 + eta) ** 2) / nw
    
    # Compute the threshold current
    Ith = injected_current_density * effective_area
    
    return Ith


try:
    targets = process_hdf5_to_tuple('42.3', 4)
    target = targets[0]
    assert (threshold_current(1, 0.02, 20, 0.0001, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4)>10**50) == target

    target = targets[1]
    assert np.allclose(threshold_current(10, 0.1, 20, 0.1, 0.3, 0.3, 3000, 200, 0.6, 2*10**-4), target)

    target = targets[2]
    assert np.allclose(threshold_current(1, 0.02, 20, 0.1, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4), target)

    target = targets[3]
    assert np.allclose(threshold_current(5, 0.02, 20, 0.1, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e