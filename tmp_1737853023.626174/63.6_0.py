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
    if N_t > 0:  # Check if there are time steps to avoid IndexError
        V[-1, :] = (p[-1] - strike * np.exp(-r * (1 - T))).flatten()

    return V


# Background: In the finite difference method for solving the Black-Scholes equation, we use a tri-diagonal matrix to
# represent the discretized version of the partial differential equation. This matrix is used to update the option prices
# at each time step. The matrix is constructed based on the coefficients derived from the Black-Scholes equation, which
# include terms for the risk-free interest rate (r) and the volatility of the underlying asset (sig). The matrix is
# tri-diagonal because each price point in the grid is only directly influenced by its immediate neighbors in the
# discretized equation. The main diagonal and the two adjacent diagonals are populated with coefficients that account
# for the changes in option price due to time decay, volatility, and the risk-free rate.



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

    if N_p < 3:
        raise ValueError("Number of grid points N_p must be at least 3.")
    if dp <= 0:
        raise ValueError("Price step dp must be positive.")
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")
    if sig <= 0:
        raise ValueError("Volatility sig must be positive.")

    # Calculate coefficients for the tri-diagonal matrix
    alpha = 0.5 * dt * ((sig**2) * np.arange(1, N_p-1)**2 - r * np.arange(1, N_p-1))
    beta = 1 - dt * ((sig**2) * np.arange(1, N_p-1)**2 + r)
    gamma = 0.5 * dt * ((sig**2) * np.arange(1, N_p-1)**2 + r * np.arange(1, N_p-1))

    # Create the tri-diagonal matrix
    diagonals = [alpha[1:], beta, gamma[:-1]]
    D = sparse.diags(diagonals, offsets=[-1, 0, 1], shape=(N_p-2, N_p-2), format='csr')

    return D


# Background: The forward iteration process in the finite difference method for solving the Black-Scholes equation
# involves using the recursive relation matrix (tri-diagonal matrix) to update the option prices at each time step.
# Starting from the known option prices at expiration (final time step), we iteratively compute the option prices
# at earlier time steps by solving a system of linear equations. This is done by multiplying the current option price
# vector by the tri-diagonal matrix to obtain the option prices at the next earlier time step. The process continues
# until we reach the initial time step, resulting in a complete 2D array of option prices across all time and price
# grid points.



def forward_iteration(V, D, N_p, N_t, r, sig, dp, dt):
    '''Performs the forward iteration to solve for option prices at earlier times.
    Inputs:
    V: A 2D array representing the grid for the option's value at different times and prices. Shape: N_p x N_t (float)
    D: The tri-diagonal matrix constructed for the finite difference method. Shape: (N_p-2) x (N_p-2) (float)
    N_p: The number of grid points in the price direction. (int)
    N_t: The number of grid points in the time direction. (int)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    Outputs:
    V: Updated option value grid after performing forward iteration. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''

    # Iterate backwards in time from the second last time step to the first
    for j in range(N_t - 2, -1, -1):
        # Solve the system of equations for the current time step
        # V[1:-1, j] represents the option prices at the current time step excluding boundaries
        # V[1:-1, j+1] represents the option prices at the next time step excluding boundaries
        V[1:-1, j] = spsolve(D, V[1:-1, j+1])

    return V


# Background: The finite difference method is a numerical technique used to solve differential equations by approximating
# them with difference equations. In the context of the Black-Scholes equation for option pricing, this method involves
# discretizing the continuous variables (price and time) into a grid and iteratively solving for option prices at each
# grid point. The process involves setting up a grid, applying boundary conditions, constructing a recursive matrix, and
# performing forward iteration to compute option prices at each time step. This function combines all these steps to
# provide a complete solution for pricing a European call option using the finite difference method.

def price_option(price_step, time_step, strike, r, sig, max_price, min_price):
    '''Prices a European call option using the finite difference method.
    Inputs:
    price_step: The number of steps or intervals in the price direction. = N_p (int)
    time_step: The number of steps or intervals in the time direction. = N_t (int)
    strike: The strike price of the European call option. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    max_price: we can't compute infinity as bound, so set a max bound. 5 * strike price is generous (float)
    min_price: avoiding 0 as a bound due to numerical instability, (1/5) * strike price is generous (float)
    Outputs:
    V: A 2D array representing the grid for the option's value. Shape: N_p x N_t where N_p is number of price grid, and N_t is number of time grid
    '''




    # Validate input parameters
    if not isinstance(price_step, int) or not isinstance(time_step, int):
        raise TypeError("price_step and time_step must be integers.")
    if price_step <= 0 or time_step <= 0 or strike <= 0 or max_price <= 0 or min_price <= 0:
        raise ValueError("All input parameters must be positive and non-zero.")
    if sig <= 0:
        raise ValueError("Volatility (sig) must be positive.")
    if max_price <= min_price:
        raise ValueError("max_price must be greater than min_price.")

    # Initialize the grid
    p, dp, T, dt = initialize_grid(price_step, time_step, strike, max_price, min_price)

    # Apply boundary conditions
    V = apply_boundary_conditions(price_step, time_step, p, T, strike, r, sig)

    # Construct the recursive matrix
    D = construct_matrix(price_step, dp, dt, r, sig)

    # Perform forward iteration to solve for option prices
    V = forward_iteration(V, D, price_step, time_step, r, sig, dp, dt)

    return V



# Background: To determine the price of a European call option at a specific time before expiration using the finite 
# difference method, we need to first compute the option prices across the entire grid of price and time. Once we have 
# the complete grid of option prices, we can extract the option price at the desired time and stock price. The finite 
# difference method involves setting up a grid, applying boundary conditions, constructing a recursive matrix, and 
# performing forward iteration to compute option prices at each grid point. The specific option price at a given time 
# and stock price is then interpolated from the computed grid.

def price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0):
    '''Prices a European call option using the finite difference method.
    Inputs:
    price_step: The number of steps or intervals in the price direction. = N_p (int)
    time_step: The number of steps or intervals in the time direction. = N_t (int)
    strike: The strike price of the European call option. (float)
    r: The risk-free interest rate.(float)
    sig: The volatility of the underlying asset. (float)
    max_price: we can't compute infinity as bound, so set a max bound. 5 * strike price is generous (float)
    min_price: avoiding 0 as a bound due to numerical instability, (1/5) * strike price is generous (float)
    t : time percentage elapsed toward expiration, ex. 0.5 means 50% * time_step, 0 <= t < 1 (float)
    S0 : price of stock at time t*time_step (float)
    Outputs:
    Price of the option at time t * time_step
    '''




    # Initialize the grid
    p, dp, T, dt = initialize_grid(price_step, time_step, strike, max_price, min_price)

    # Apply boundary conditions
    V = apply_boundary_conditions(price_step, time_step, p, T, strike, r, sig)

    # Construct the recursive matrix
    D = construct_matrix(price_step, dp, dt, r, sig)

    # Perform forward iteration to solve for option prices
    V = forward_iteration(V, D, price_step, time_step, r, sig, dp, dt)

    # Find the index corresponding to the given time t
    time_index = int(t * (time_step - 1))

    # Interpolate the option price at the given stock price S0
    if S0 <= min_price:
        return V[0, time_index]
    elif S0 >= max_price:
        return V[-1, time_index]
    else:
        # Find the indices surrounding S0
        lower_index = int((S0 - min_price) / dp)
        upper_index = lower_index + 1

        # Linear interpolation
        S_lower = p[lower_index, 0]
        S_upper = p[upper_index, 0]
        V_lower = V[lower_index, time_index]
        V_upper = V[upper_index, time_index]

        # Interpolated option price
        Price = V_lower + (S0 - S_lower) * (V_upper - V_lower) / (S_upper - S_lower)

    return Price

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.6', 3)
target = targets[0]

price_step = 3000  # Number of price steps
time_step = 3000   # Number of time steps
strike = 1000      # Strike price of the option
r = 0.05          # Risk-free interest rate
sig = 1         # Volatility of the underlying asset
S0 = 100          # Initial stock price
max_price = 5 * strike # maximum bound on price grid
min_price = (1/5) * strike # minimum bound on price grid
t = 0
assert np.allclose(price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0), target)
target = targets[1]

price_step = 3000  # Number of price steps
time_step = 3000   # Number of time steps
strike = 1000      # Strike price of the option
r = 0.05          # Risk-free interest rate
sig = 1         # Volatility of the underlying asset
S0 = 500          # Initial stock price
max_price = 5 * strike # maximum bound on price grid
min_price = (1/5) * strike # minimum bound on price grid
t = 0.5
assert np.allclose(price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0), target)
target = targets[2]

price_step = 3000  # Number of price steps
time_step = 3000   # Number of time steps
strike = 3000      # Strike price of the option
r = 0.2          # Risk-free interest rate
sig = 1         # Volatility of the underlying asset
S0 = 1000        # Initial stock price
max_price = 5 * strike # maximum bound on price grid
min_price = (1/5) * strike # minimum bound on price grid
t = 0.5
assert np.allclose(price_option_of_time(price_step, time_step, strike, r, sig, max_price, min_price, t, S0), target)
