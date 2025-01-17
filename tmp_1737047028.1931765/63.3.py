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


# Background: In the context of the Black-Scholes equation, boundary conditions are essential for solving the partial 
# differential equation using numerical methods like the finite difference method. For a European call option, the 
# boundary conditions are defined as follows:
# 1. At expiration (T = 0), the option value is max(S - K, 0), where S is the stock price and K is the strike price.
# 2. As the stock price approaches zero, the option value approaches zero.
# 3. As the stock price becomes very large, the option value approaches the stock price minus the present value of the 
#    strike price, i.e., S - K * exp(-r * T), where r is the risk-free interest rate.
# These conditions ensure that the option pricing model behaves correctly at the boundaries of the grid.




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

    # Apply the terminal condition at T = 0 (i.e., at expiration)
    # V(S, 0) = max(S - K, 0)
    V[:, 0] = np.maximum(p.flatten() - strike, 0)

    # Apply the boundary condition as S -> 0, V -> 0
    # This is already handled by initializing V to zeros

    # Apply the boundary condition as S -> infinity, V -> S - K * exp(-r * T)
    # For large S, the option value approaches intrinsic value minus the discounted strike price
    V[-1, :] = p[-1] - strike * np.exp(-r * T.flatten())

    return V



# Background: In the finite difference method for solving the Black-Scholes equation, we use a tri-diagonal matrix to
# represent the discretized version of the partial differential equation. This matrix is used to iteratively compute
# the option prices at each time step. The matrix is constructed based on the coefficients derived from the Black-Scholes
# equation, which include the risk-free interest rate (r) and the volatility of the underlying asset (sig). The matrix
# is tri-diagonal because it involves the current price, the price one step below, and the price one step above in the
# grid. The finite difference method uses this matrix to update the option prices from one time step to the next.



def construct_matrix(N_p, dp, dt, r, sig):
    '''Constructs the tri-diagonal matrix for the finite difference method.
    Inputs:
    N_p: The number of grid points in the price direction. (int)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    Outputs:
    D: The tri-diagonal matrix constructed for the finite difference method. Shape: (N_p-2)x(N_p-2) where N_p is number of price grid, and N_t is number of time grid minus 2 due to boundary conditions
    '''

    # Calculate coefficients for the finite difference method
    alpha = 0.5 * dt * ((sig**2) * np.arange(1, N_p-1)**2 - r * np.arange(1, N_p-1))
    beta = 1 - dt * ((sig**2) * np.arange(1, N_p-1)**2 + r)
    gamma = 0.5 * dt * ((sig**2) * np.arange(1, N_p-1)**2 + r * np.arange(1, N_p-1))

    # Create the tri-diagonal matrix
    diagonals = [alpha[1:], beta, gamma[:-1]]
    D = sparse.diags(diagonals, offsets=[-1, 0, 1], shape=(N_p-2, N_p-2), format='csr')

    return D


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('63.3', 3)
target = targets[0]

N_p=1000
r=0.2
sig=4
dt = 1
dp =1
assert np.allclose(construct_matrix(N_p,dp,dt,r,sig).toarray(), target)
target = targets[1]

N_p=3000
r=0.1
sig=10
dt = 1
dp =1
assert np.allclose(construct_matrix(N_p,dp,dt,r,sig).toarray(), target)
target = targets[2]

N_p=1000
r=0.5
sig=1
dt = 1
dp =1
assert np.allclose(construct_matrix(N_p,dp,dt,r,sig).toarray(), target)
