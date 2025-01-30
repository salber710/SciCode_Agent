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

    # Initialize the option value grid with random small positive values to distinguish
    # from zero or other initializations
    V = np.random.rand(N_p, N_t) * 1e-3

    # Boundary condition at maturity (T = 0)
    # V(S, 0) = max(S - K, 0)
    for i in range(N_p):
        V[i, 0] = max(p[i] - strike, 0)

    # Boundary condition as S approaches 0 over time
    # V(0, t) = 0 for all t
    V[0, :] = 0

    # Boundary condition as S approaches infinity
    # Use a novel boundary condition incorporating hyperbolic tangent
    for j in range(N_t):
        V[-1, j] = (p[-1] - strike * np.exp(-r * T[j])) * np.tanh(sig * T[j] / (1 + T[j]))

    # Fill the grid using a logarithmic decay approach
    for j in range(1, N_t):
        for i in range(1, N_p - 1):
            intrinsic_value = max(p[i] - strike, 0)
            decay_factor = np.exp(-r * T[j]) * np.log(1 + sig * T[j])
            V[i, j] = intrinsic_value * decay_factor

    return V



# Background: The finite difference method is a computational technique used to solve differential equations by approximating them with difference equations. In the context of option pricing, the Black-Scholes partial differential equation (PDE) can be discretized using finite differences. The resulting equations can be represented in a matrix form, where the unknown option prices at the next time step can be solved using a tri-diagonal matrix. This matrix captures the relationships between option prices at adjacent grid points and is constructed based on the coefficients derived from the Black-Scholes PDE. The matrix is typically tri-diagonal because each price point in the grid is influenced by itself and its immediate neighbors, which leads to non-zero coefficients on the main diagonal and the diagonals directly above and below the main diagonal.

def construct_matrix(N_p, dp, dt, r, sig):
    '''Constructs the tri-diagonal matrix for the finite difference method.
    Inputs:
    N_p: The number of grid points in the price direction. (int)
    dp: The spacing between adjacent price grid points. (float)
    dt: The spacing between adjacent time grid points. (float)
    r: The risk-free interest rate. (float)
    sig: The volatility of the underlying asset. (float)
    Outputs:
    D: The tri-diagonal matrix constructed for the finite difference method. Shape: (N_p-2)x(N_p-2)
    '''
    
    # Pre-calculate constants used in the matrix
    alpha = 0.5 * dt * (sig**2 * np.arange(1, N_p-1)**2 - r * np.arange(1, N_p-1))
    beta = 1 - dt * (sig**2 * np.arange(1, N_p-1)**2 + r)
    gamma = 0.5 * dt * (sig**2 * np.arange(1, N_p-1)**2 + r * np.arange(1, N_p-1))

    # Create three diagonals for the tri-diagonal matrix
    lower_diag = -alpha[1:]  # Lower diagonal (below the main diagonal)
    main_diag = 1 - beta     # Main diagonal
    upper_diag = -gamma[:-1] # Upper diagonal (above the main diagonal)

    # Construct the sparse matrix
    D = sparse.diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], shape=(N_p-2, N_p-2)).tocsc()

    return D


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e