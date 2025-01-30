import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Background: The Black-Scholes equation is a partial differential equation used to model the price of options over time. 
# To solve this equation numerically, we can use the finite difference method, which involves discretizing the continuous 
# variables (price and time) into a grid. The grid is defined by a set of points in the price and time dimensions. 
# The price grid is created between a minimum and maximum price, and the time grid is created between the start and end 
# of the option's life (usually normalized to 0 to 1). The step sizes in both dimensions (dp for price and dt for time) 
# are determined by the number of intervals (price_step and time_step) specified. This setup is crucial for applying 
# numerical methods to approximate the solution to the Black-Scholes equation.

def initialize_grid(price_step, time_step, strike, max_price, min_price):
    '''Initializes the grid for pricing a European call option.
    Inputs:
    price_step: The number of steps or intervals in the price direction. (int)
    time_step: The number of steps or intervals in the time direction. (int)
    strike: The strike price of the European call option. (float)
    max_price: we can't compute infinity as bound, so set a max bound. 5 * strike price is generous (float)
    min_price: avoiding 0 as a bound due to numerical instability, (1/5) * strike price is generous (float)
    Outputs:
    p: An array containing the grid points for prices. It is calculated using np.linspace function between p_min and p_max.  shape: price_step * 1
    dp: The spacing between adjacent price grid points. (float)
    T: An array containing the grid points for time. It is calculated using np.linspace function between 0 and 1. shape: time_step * 1
    dt: The spacing between adjacent time grid points. (float)
    '''


    if price_step < 1 or time_step < 1:
        raise ValueError("price_step and time_step must be at least 1")
    if strike < 0:
        raise ValueError("strike price must be non-negative")
    if max_price < min_price:
        raise ValueError("max_price must be greater than or equal to min_price")

    # Create the price grid using linspace from min_price to max_price
    p = np.linspace(min_price, max_price, price_step).reshape(-1, 1)
    # Calculate the price step size
    dp = (max_price - min_price) / (price_step - 1) if price_step > 1 else 0

    # Create the time grid using linspace from 0 to 1
    T = np.linspace(0, 1, time_step).reshape(-1, 1)
    # Calculate the time step size
    dt = 1 / (time_step - 1) if time_step > 1 else 0

    return p, dp, T, dt



# Background: In the context of the Black-Scholes equation, boundary conditions are essential for solving the partial 
# differential equation numerically. For a European call option, the boundary conditions are defined as follows:
# 1. At expiration (T = 1), the option value is max(S - K, 0), where S is the stock price and K is the strike price.
# 2. As the stock price approaches zero, the option value approaches zero.
# 3. As the stock price becomes very large, the option value approaches the stock price minus the present value of the 
#    strike price, i.e., S - K * exp(-r * (T-t)), where r is the risk-free interest rate and t is the current time.
# These conditions help in setting up the initial and boundary values for the finite difference grid, which are crucial 
# for solving the Black-Scholes equation using numerical methods.


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

    # Initialize the option price grid V with zeros
    V = np.zeros((N_p, N_t))

    # Apply the terminal condition at T = 1 (last column in V)
    V[:, -1] = np.maximum(p - strike, 0).flatten()

    # Apply the boundary condition for S -> 0 (first row in V)
    V[0, :] = 0

    # Apply the boundary condition for S -> infinity (last row in V)
    V[-1, :] = (p[-1] - strike * np.exp(-r * (1 - T))).flatten()

    return V

from scicode.parse.parse import process_hdf5_to_tuple
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
