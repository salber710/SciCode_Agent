from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fft, fftfreq


def gillespie_step(prey, predator, alpha, beta, gamma):
    # Define event rates using a list for direct access and manipulation
    rates = [alpha * prey, beta * prey * predator, gamma * predator]
    
    # Calculate the total rate
    total_rate = sum(rates)
    
    # If no events can occur, return None
    if total_rate == 0:
        return None, prey, predator, None
    
    # Sample the time step from an exponential distribution
    time_step = np.random.exponential(1 / total_rate)
    
    # Determine which event occurs using a direct index method
    random_value = np.random.uniform(0, total_rate)
    event_index = 0
    cumulative_rate = 0
    for rate in rates:
        cumulative_rate += rate
        if random_value < cumulative_rate:
            break
        event_index += 1
    
    # Update populations based on the event
    if event_index == 0:
        prey += 1
        event = "prey_birth"
    elif event_index == 1:
        prey -= 1
        predator += 1
        event = "predation"
    else:
        predator -= 1
        event = "predator_death"
    
    return time_step, prey, predator, event


def evolve_LV(prey, predator, alpha, beta, gamma, T):


    # Initialize variables
    time_cor = [0]
    prey_evol = [prey]
    predator_evol = [predator]
    current_time = 0

    # Define the rate function for each event
    def rate_prey_birth(prey):
        return alpha * prey

    def rate_predation(prey, predator):
        return beta * prey * predator

    def rate_predator_death(predator):
        return gamma * predator

    # Simulation loop using a deterministic approach for event timing
    while current_time < T and prey > 0 and predator > 0:
        rates = [rate_prey_birth(prey), rate_predation(prey, predator), rate_predator_death(predator)]
        total_rate = sum(rates)

        if total_rate == 0:
            break

        # Deterministic time to next event
        time_to_event = 1 / total_rate
        current_time += time_to_event
        if current_time > T:
            break

        # Select event deterministically based on the smallest rate
        min_rate_index = np.argmin(rates)
        event = ['prey_birth', 'predation', 'predator_death'][min_rate_index]

        # Update populations based on the event
        if event == 'prey_birth':
            prey += 1
        elif event == 'predation':
            prey -= 1
            predator += 1
        elif event == 'predator_death':
            predator -= 1

        # Record the state
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)

    # Determine the ecological event
    if prey > 0 and predator > 0:
        eco_event = "coexistence"
    elif prey > 0 and predator == 0:
        eco_event = "predator extinction"
    else:
        eco_event = "mutual extinction"

    return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event




def spectral_periodicity(t, population):
    '''Estimate the periodicity of population with uneven time step and stochasticity.
    Input:
    t: time coordinates of population evolution, 1D array of floats
    population: evolution history of population of some species, 1D array of floats (same size as t)
    Output:
    periodicity: estimated periodicity, float rounded up to one decimal point.
    '''
    
    # Define a sinusoidal function to fit the data
    def sinusoidal(x, A, B, omega, phi):
        return A * np.sin(omega * x + phi) + B

    # Initial guess for parameters: Amplitude, Offset, Angular frequency, Phase shift
    initial_guess = [np.std(population), np.mean(population), 2 * np.pi / (t[-1] - t[0]), 0]

    # Fit the sinusoidal function to the data
    params, _ = curve_fit(sinusoidal, t, population, p0=initial_guess)

    # Extract the frequency from the fitted parameters
    omega = params[2]
    frequency = omega / (2 * np.pi)

    # Calculate the period as the inverse of the frequency
    periodicity = 1 / frequency

    # Round the periodicity to one decimal point
    return round(periodicity, 1)



# Background: The Lotka-Volterra equations describe the dynamics of biological systems in which two species interact, predator and prey. 
# The Gillespie algorithm is a stochastic simulation method used to simulate the time evolution of a system with discrete events. 
# It is particularly useful for systems with small populations where stochastic effects are significant. 
# In this context, we simulate the predator-prey dynamics using the Gillespie algorithm, recording the populations over time. 
# We then analyze the resulting time series to determine the ecological event that occurs: "coexistence", "predator extinction", or "mutual extinction". 
# If coexistence is observed, we estimate the periodicity of the population oscillations using Fourier analysis.




def predator_prey(prey, predator, alpha, beta, gamma, T):
    '''Simulate the predator-prey dynamics using the Gillespie simulation algorithm.
    Records the populations of prey and predators and the times at which changes occur.
    Analyze the ecological phenomenon happens in the system.
    Input:
    prey: initial population of prey, integer
    predator: initial population of predators, integer
    alpha: prey birth rate, float
    beta: predation rate, float
    gamma: predator death rate, float
    T: total time of the simulation, float
    Output:
    time_cor: time coordinates of population evolution, 1D array of floats
    prey_evol: evolution history of prey population, 1D array of floats (same size as time_cor)
    predator_evol: evolution history of predator population, 1D array of floats (same size as time_cor)
    eco_event: A string describing the ecological event ("coexistence", "predator extinction", or "mutual extinction").
    prey_period: estimated periodicity of prey population, float rounded up to one decimal point; 0.0 if no coexistence
    predator_period: estimated periodicity of predator population, float rounded up to one decimal point; 0.0 if no coexistence
    '''

    # Initialize variables
    time_cor = [0]
    prey_evol = [prey]
    predator_evol = [predator]
    current_time = 0

    # Simulation loop using the Gillespie algorithm
    while current_time < T and prey > 0 and predator > 0:
        # Define event rates
        rates = [alpha * prey, beta * prey * predator, gamma * predator]
        total_rate = sum(rates)

        if total_rate == 0:
            break

        # Sample the time step from an exponential distribution
        time_step = np.random.exponential(1 / total_rate)
        current_time += time_step
        if current_time > T:
            break

        # Determine which event occurs
        random_value = np.random.uniform(0, total_rate)
        event_index = 0
        cumulative_rate = 0
        for rate in rates:
            cumulative_rate += rate
            if random_value < cumulative_rate:
                break
            event_index += 1

        # Update populations based on the event
        if event_index == 0:
            prey += 1
        elif event_index == 1:
            prey -= 1
            predator += 1
        else:
            predator -= 1

        # Record the state
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)

    # Determine the ecological event
    if prey > 0 and predator > 0:
        eco_event = "coexistence"
    elif prey > 0 and predator == 0:
        eco_event = "predator extinction"
    else:
        eco_event = "mutual extinction"

    # Function to estimate periodicity using Fourier Transform
    def estimate_periodicity(t, population):
        if len(t) < 2:
            return 0.0
        # Interpolate to get evenly spaced time points
        interp_func = interp1d(t, population, kind='linear', fill_value='extrapolate')
        t_even = np.linspace(t[0], t[-1], num=len(t))
        population_even = interp_func(t_even)
        
        # Perform Fourier Transform
        fft_vals = fft(population_even)
        freqs = fftfreq(len(t_even), d=(t_even[1] - t_even[0]))
        
        # Find the peak frequency
        positive_freqs = freqs[freqs > 0]
        positive_fft_vals = np.abs(fft_vals[freqs > 0])
        peak_freq = positive_freqs[np.argmax(positive_fft_vals)]
        
        # Calculate the period
        if peak_freq > 0:
            period = 1 / peak_freq
            return round(period, 1)
        else:
            return 0.0

    # Calculate periodicity if coexistence
    if eco_event == "coexistence":
        prey_period = estimate_periodicity(time_cor, prey_evol)
        predator_period = estimate_periodicity(time_cor, predator_evol)
    else:
        prey_period = 0.0
        predator_period = 0.0

    return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event, prey_period, predator_period


try:
    targets = process_hdf5_to_tuple('53.4', 3)
    target = targets[0]
    np.random.seed(2)
    prey, predator = 200, 200
    alpha, beta, gamma = 2., 0.01, 3.
    T = 20.
    time_cor, prey_evol, predator_evol, eco_event, prey_period, predator_period = predator_prey(prey, predator, alpha, beta, gamma, T)
    a, b, c, d, e, f = target
    assert np.allclose(time_cor, a) and np.allclose(prey_evol, b) and np.allclose(predator_evol, c) and eco_event == d and prey_period == e and predator_period == f

    target = targets[1]
    np.random.seed(1)
    prey, predator = 100, 20
    alpha, beta, gamma = 3., 0.05, 1.
    T = 10.
    time_cor, prey_evol, predator_evol, eco_event, prey_period, predator_period = predator_prey(prey, predator, alpha, beta, gamma, T)
    a, b, c, d, e, f = target
    assert np.allclose(time_cor, a) and np.allclose(prey_evol, b) and np.allclose(predator_evol, c) and eco_event == d and prey_period == e and predator_period == f

    target = targets[2]
    np.random.seed(3)
    prey, predator = 100, 10
    alpha, beta, gamma = 1., 0.005, 5.
    T = 10.
    time_cor, prey_evol, predator_evol, eco_event, prey_period, predator_period = predator_prey(prey, predator, alpha, beta, gamma, T)
    a, b, c, d, e, f = target
    assert np.allclose(time_cor, a) and np.allclose(prey_evol, b) and np.allclose(predator_evol, c) and eco_event == d and prey_period == e and predator_period == f

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e