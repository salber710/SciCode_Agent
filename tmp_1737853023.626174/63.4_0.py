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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('63.4', 3)
target = targets[0]

N_p=1000
N_t=2000
r=0.05
sig=1
dt = 1
dp =1
strike = 500
min_price = 100
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
V = apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig)
D = construct_matrix(N_p,dp,dt,r,sig)
assert np.allclose(forward_iteration(V, D, N_p, N_t, r, sig, dp, dt), target)
target = targets[1]

N_p=2000
N_t=3000
r=0.1
sig=2
dt = 1
dp =1
strike = 200
min_price = 100
max_price = 2500
p, dp, T, dt = initialize_grid(N_p,N_t,strike, min_price, max_price)
V = apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig)
D = construct_matrix(N_p,dp,dt,r,sig)
assert np.allclose(forward_iteration(V, D, N_p, N_t, r, sig, dp, dt), target)
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
V = apply_boundary_conditions(N_p,N_t,p,T,strike,r,sig)
D = construct_matrix(N_p,dp,dt,r,sig)
assert np.allclose(forward_iteration(V, D, N_p, N_t, r, sig, dp, dt), target)
