import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



# Background: The Black-Scholes equation is a partial differential equation used to model the price of options over time. 
# To solve this equation numerically, we can use the finite difference method, which involves discretizing the continuous 
# variables (price and time) into a grid. The price grid represents different possible stock prices, while the time grid 
# represents different time points leading up to the option's expiration. The step sizes (dp and dt) are the intervals 
# between consecutive points in the price and time grids, respectively. These grids allow us to approximate the derivatives 
# in the Black-Scholes equation and solve it iteratively.

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


    # Create the price grid using np.linspace from min_price to max_price
    p = np.linspace(min_price, max_price, price_step).reshape(-1, 1)
    # Calculate the price step size
    dp = (max_price - min_price) / (price_step - 1)

    # Create the time grid using np.linspace from 0 to 1
    T = np.linspace(0, 1, time_step).reshape(-1, 1)
    # Calculate the time step size
    dt = 1 / (time_step - 1)

    return p, dp, T, dt


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('63.1', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
price_step = 5000
time_step = 2000
strike = 100
min_price = 20
max_price = 500
assert cmp_tuple_or_list(initialize_grid(price_step,time_step,strike, min_price, max_price), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
price_step = 3000
time_step = 2000
strike = 500
min_price = 100
max_price = 2500
assert cmp_tuple_or_list(initialize_grid(price_step,time_step,strike, min_price, max_price), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
price_step = 3000
time_step = 2000
strike = 50
min_price = 10
max_price = 250
assert cmp_tuple_or_list(initialize_grid(price_step,time_step,strike, min_price, max_price), target)
