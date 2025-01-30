from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def initialize_grid(price_step, time_step, strike, max_price, min_price):
    '''Initializes the grid for pricing a European call option using a random spacing for the grid points.
    Inputs:
    price_step: The number of steps or intervals in the price direction. (int)
    time_step: The number of steps or intervals in the time direction. (int)
    strike: The strike price of the European call option. (float)
    max_price: we can't compute infinity as bound, so set a max bound. 5 * strike price is generous (float)
    min_price: avoiding 0 as a bound due to numerical instability, (1/5) * strike price is generous (float)
    Outputs:
    p: An array containing the grid points for prices using random uniform spacing. shape: price_step * 1
    dp: The average spacing between adjacent price grid points. (float)
    T: An array containing the grid points for time using random uniform spacing. shape: time_step * 1
    dt: The average spacing between adjacent time grid points. (float)
    '''
    
    # Random uniform spacing for the price grid
    p_random = np.random.uniform(min_price, max_price, price_step)
    p_random.sort()  # Ensure the points are sorted
    p = p_random.reshape(price_step, 1)
    dp = np.mean(np.diff(p, axis=0))

    # Random uniform spacing for the time grid
    T_random = np.random.uniform(0, 1, time_step)
    T_random.sort()  # Ensure the points are sorted
    T = T_random.reshape(time_step, 1)
    dt = np.mean(np.diff(T, axis=0))
    
    return p, dp, T, dt



# Background: In the context of the Black-Scholes equation for pricing European call options, boundary conditions are crucial for solving the partial differential equation via numerical methods such as the finite-difference method. 
# For a European call option, the boundary conditions are typically set as follows:
# 1. At maturity (T = 0), the option payoff is max(S - K, 0) for each stock price S, where K is the strike price.
# 2. As the stock price approaches zero, the option value should be zero (since the stock is worthless).
# 3. As the stock price becomes very large (theoretically infinite), the option value approaches S - K, since the option will be exercised in this case.

def apply_boundary_conditions(N_p, N_t, p, T, strike, r, sig):
    '''Applies the boundary conditions to the grid.
    Inputs:
    N_p: The number of grid points in the price direction. = price_step (int)
    N_t: The number of grid points in the time direction. = time_step (int)
    p: An array containing the grid points for prices. (shape = 1 * N_p , (float))
    T: An array containing the grid points for time. (shape = 1 * N_t , (float))
    strike: The strike price of the European call option. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying stock. (float)
    Outputs:
    V: A 2D array representing the grid for the option's value after applying boundary conditions. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''

    # Initialize the option value grid with zeros
    V = np.zeros((N_p, N_t))

    # Apply boundary condition at maturity (T = 0)
    # V(S, 0) = max(S - K, 0)
    for i in range(N_p):
        V[i, 0] = max(p[i] - strike, 0)

    # Apply boundary condition as S approaches 0
    # V(0, t) = 0 for all t
    # This is already set to zero by initialization, but for clarity:
    V[0, :] = 0

    # Apply boundary condition as S approaches infinity
    # V(S, t) = S - K * exp(-r * t) for large S
    for j in range(N_t):
        V[-1, j] = p[-1] - strike * np.exp(-r * T[j])

    return V


try:
    targets = process_hdf5_to_tuple('63.2', 3)
    target = targets[0]
    N_p=1000
    N_t=2000
    r=0.02
    sig=2
    dt = 1
    dp =1
    strike = 1000
    min_price = 300
    max_price = 2500
    p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
    assert np.allclose(apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig), target)

    target = targets[1]
    N_p=4000
    N_t=4000
    r=0.2
    sig=1
    dt = 1
    dp =1
    strike = 1000
    min_price = 100
    max_price = 2500
    p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
    assert np.allclose(apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig), target)

    target = targets[2]
    N_p=1000
    N_t=2000
    r=0.5
    sig=1
    dt = 1
    dp =1
    strike = 1000
    min_price = 100
    max_price = 2500
    p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
    assert np.allclose(apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e