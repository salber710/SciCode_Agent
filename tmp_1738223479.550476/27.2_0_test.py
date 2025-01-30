from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def Fermi(N_A, N_D, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_A: float, doping concentration in p-type region # cm^{-3}
    N_D: float, doping concentration in n-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''

    # Thermal potential at room temperature in volts
    V_T = 0.0259

    # Calculate built-in biases using Simpson's Rule for integral approximation
    def simpsons_log_approximation(x, n_segments=1000):
        """Approximate natural log(x) using Simpson's Rule."""
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        
        f = lambda t: 1 / t
        a, b = 1, x
        h = (b - a) / n_segments
        integral = f(a) + f(b)
        
        for i in range(1, n_segments, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n_segments-1, 2):
            integral += 2 * f(a + i * h)
        
        integral *= h / 3
        return integral

    # Calculate the built-in biases using the Simpson's Rule approximation
    phi_p = V_T * simpsons_log_approximation(N_A / n_i)
    phi_n = V_T * simpsons_log_approximation(N_D / n_i)

    return phi_p, phi_n



# Background: 
# The capacitance of a p-i-n diode is influenced by the intrinsic region's width and the permittivity of the material.
# In a p-i-n diode, the intrinsic region (i-region) is undoped and acts as a dielectric between the p-type and n-type regions.
# The capacitance (C) of the diode can be computed using the formula: 
# C = (ε_r * ε_0 * A) / x_i, where ε_r is the relative permittivity, ε_0 is the vacuum permittivity, A is the detector area, 
# and x_i is the thickness of the intrinsic layer. 
# This formula is derived from the basic capacitance formula C = εA/d for a parallel plate capacitor, where d is the separation between the plates.

def capacitance(xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the capacitance of a p-i-n diode.
    Input:
    xi (float): Width of the intrinsic region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density of the material # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the p-i-n diode (V).
    Output:
    C (float): Capacitance of the p-i-n diode (F).
    '''
    
    # Convert intrinsic layer thickness and area from micrometers to meters
    xi_m = xi * 1e-6
    A_m2 = A * 1e-12
    
    # Vacuum permittivity in F/m
    epsilon_0 = 8.854e-12
    
    # Calculate the capacitance using the capacitance formula for a parallel plate capacitor
    C = (es * epsilon_0 * A_m2) / xi_m
    
    return C


try:
    targets = process_hdf5_to_tuple('27.2', 3)
    target = targets[0]
    assert np.allclose(capacitance(5, 1000, 1e10, 1e8,1.8e6,13, 0)*10**15, target)

    target = targets[1]
    assert np.allclose(capacitance(0.1, 700, 1e19, 1e17,1.8e6,13, 0)*10**15, target)

    target = targets[2]
    assert np.allclose(capacitance(0.5, 1000, 1e19, 1e17,1.8e6,13, -3)*10**15, target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e