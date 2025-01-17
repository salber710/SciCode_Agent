import numpy as np

# Background: 
# The task is to compute the initial condition as a cell-averaged approximation using the three-point Gauss quadrature rule.
# Gauss quadrature is a numerical integration method that approximates the integral of a function. The three-point Gauss quadrature
# rule uses three specific points and weights to approximate the integral over an interval. For the interval [-1, 1], the points
# are x1 = -sqrt(3/5), x2 = 0, x3 = sqrt(3/5) with weights w1 = w3 = 5/9 and w2 = 8/9. To apply this to an arbitrary interval [a, b],
# we transform the points and weights accordingly. The initial condition function u_0 is piecewise defined, and we need to compute
# the cell-averaged values over each subinterval of the domain [-pi/2, pi/2] divided into n-1 cells.


def make_IC(n):
    '''The function computes the initial condition mentioned above
    Inputs:
    n  : number of grid points, integer
    Outputs:
    v  : cell averaged approximation of initial condition, 1d array size n-1
    '''
    # Define the domain
    x_start = -np.pi / 2
    x_end = np.pi / 2
    
    # Calculate the cell width
    dx = (x_end - x_start) / (n - 1)
    
    # Gauss quadrature points and weights for the interval [-1, 1]
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    # Initialize the array for cell-averaged values
    v = np.zeros(n - 1)
    
    # Loop over each cell
    for i in range(n - 1):
        # Calculate the left and right endpoints of the current cell
        x_left = x_start + i * dx
        x_right = x_left + dx
        
        # Transform Gauss points to the current cell
        x_gauss = 0.5 * (x_right - x_left) * gauss_points + 0.5 * (x_right + x_left)
        
        # Evaluate the initial condition at the Gauss points
        u_values = np.where(x_gauss <= 0, np.sin(x_gauss) - 1, np.sin(x_gauss) + 1)
        
        # Compute the integral using Gauss quadrature
        integral = np.sum(gauss_weights * u_values) * 0.5 * (x_right - x_left)
        
        # Compute the cell-averaged value
        v[i] = integral / dx
    
    return v



# Background: The Lax-Friedrichs method is a numerical technique used to solve hyperbolic partial differential equations.
# It is a type of finite volume method that approximates the flux at the interface between two cells. The Lax-Friedrichs
# flux is computed using the average of the fluxes from the left and right states, with a stabilization term that involves
# the maximum wave speed, denoted as alpha_LF. This stabilization term helps to ensure numerical stability by accounting
# for the maximum speed at which information can propagate through the system. The formula for the Lax-Friedrichs flux is:
# 
# flux = 0.5 * (f(uL) + f(uR)) - 0.5 * alpha_LF * (uR - uL)
#
# where f(u) is the flux function, which in the simplest linear case can be taken as f(u) = u. The parameter alpha_LF
# is chosen as the maximum wave speed in the system, which can be problem-specific.


def LaxF(uL, uR):
    '''This function computes Lax-Friedrichs numerical flux.
    Inputs: 
    uL : Cell averaged value at cell i, float
    uR : Cell averaged value at cell i+1, float
    Output: flux, float
    '''
    # Define the maximum wave speed, alpha_LF
    # For a simple linear advection problem, this can be the maximum absolute value of the wave speed.
    # Here, we assume a wave speed of 1 for simplicity, but this should be adjusted based on the problem.
    alpha_LF = 1.0
    
    # Compute the Lax-Friedrichs flux
    flux = 0.5 * (uL + uR) - 0.5 * alpha_LF * (uR - uL)
    
    return flux


from scicode.parse.parse import process_hdf5_to_tuple

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
