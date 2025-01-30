from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy as sp
import scipy.integrate as si

def eos_press_from_rho(rho, eos_Gamma, eos_kappa):
    '''This function computes pressure for a polytropic equation of state given the density 
    using a continued fraction approximation for exponentiation.
    Inputs:
    rho: the density, a float.
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    press: pressure corresponding to the given density, a float
    '''

    # Continued fraction approximation for power function
    def continued_fraction_power(base, exp, iterations=10):
        # Initialize continued fraction components
        a = [exp] * iterations
        b = [1] * iterations
        b[0] = base  # Start with base as the first b term

        # Compute continued fraction
        result = a[-1]
        for i in range(iterations - 2, -1, -1):
            result = a[i] + b[i] / result

        return base ** exp / result

    # Calculate rho to the power of eos_Gamma using continued fraction approximation
    rho_exp = continued_fraction_power(rho, eos_Gamma)

    # Calculate pressure
    press = eos_kappa * rho_exp

    return press



def eos_rho_from_press(press, eos_Gamma, eos_kappa):
    '''This function computes density for a polytropic equation of state given the pressure.
    Inputs:
    press: pressure, a float
    eos_Gamma: adiabatic exponent of the equation of state, a float
    eos_kappa: coefficient of the equation of state, a float
    Outputs:
    rho: the density corresponding to the given pressure, a float
    '''

    # Use a recursive approach to approximate the density.
    # This is distinct from iterative or direct calculations.

    def recursive_density(pressure, gamma, kappa, rho_guess, depth):
        # Base case: if depth is zero, return the current guess
        if depth == 0:
            return rho_guess
        
        # Calculate the pressure from the current guess
        press_guess = kappa * rho_guess ** gamma
        
        # Update the guess using a recursive approach
        new_rho_guess = rho_guess * (pressure / press_guess) ** (1 / gamma)
        
        # Recurse with reduced depth
        return recursive_density(pressure, gamma, kappa, new_rho_guess, depth - 1)

    # Initial guess for density
    initial_rho_guess = 1.0
    
    # Set the depth for recursion
    recursion_depth = 10
    
    # Compute the density using the recursive function
    rho = recursive_density(press, eos_Gamma, eos_kappa, initial_rho_guess, recursion_depth)
    
    return rho


try:
    targets = process_hdf5_to_tuple('58.2', 3)
    target = targets[0]
    press = 10
    eos_Gamma = 20
    eos_kappa = 30
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

    target = targets[1]
    press = 1000
    eos_Gamma = 50
    eos_kappa = 80
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

    target = targets[2]
    press = 20000
    eos_Gamma = 2.
    eos_kappa = 100.
    assert np.allclose(eos_rho_from_press(press, eos_Gamma, eos_kappa), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e