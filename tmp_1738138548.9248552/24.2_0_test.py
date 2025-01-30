from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def make_IC(n):
    x_start = -np.pi / 2
    x_end = np.pi / 2
    dx = (x_end - x_start) / (n - 1)
    
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    v = np.zeros(n - 1)
    
    for i in range(n - 1):
        a = x_start + i * dx
        b = a + dx
        mapped_points = 0.5 * (b - a) * gauss_points + 0.5 * (b + a)
        
        # Evaluate the function using numpy.piecewise
        f_values = np.piecewise(mapped_points, [mapped_points <= 0, mapped_points > 0],
                                [lambda x: np.sin(x) - 1, lambda x: np.sin(x) + 1])
        
        # Compute the integral using Gauss quadrature
        integral = np.dot(gauss_weights, f_values) * 0.5 * (b - a)
        
        # Compute the cell-averaged value
        v[i] = integral / dx
    
    return v



# Background: The Lax-Friedrichs (LF) numerical flux is a method used in the numerical solution of hyperbolic partial differential equations. 
# It is a type of finite volume method that approximates the flux at the interface between two cells. 
# The LF flux is computed using the average of the fluxes from the left and right states, with an added dissipation term to ensure stability.
# The dissipation term is proportional to the difference between the left and right states, scaled by a stability parameter, α_LF.
# The stability parameter, α_LF, is typically chosen as the maximum wave speed in the system, which ensures that the numerical scheme is stable.

def LaxF(uL, uR):
    '''This function computes Lax-Friedrich numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Define the maximum wave speed, which is used as the stability parameter α_LF
    # For simplicity, we assume a constant wave speed, which can be adjusted based on the problem context
    alpha_LF = 1.0  # This is a placeholder value; in practice, it should be the maximum wave speed

    # Compute the Lax-Friedrichs flux
    flux = 0.5 * (uL + uR) - 0.5 * alpha_LF * (uR - uL)

    return flux


try:
    targets = process_hdf5_to_tuple('24.2', 3)
    target = targets[0]
    v_i = 0
    v_i1 = 1
    assert np.allclose(LaxF(v_i,v_i1), target)

    target = targets[1]
    v_i = 3
    v_i1 = 3
    assert np.allclose(LaxF(v_i,v_i1), target)

    target = targets[2]
    v_i = 5
    v_i1 = -5
    assert np.allclose(LaxF(v_i,v_i1), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e