from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fft, fftfreq


def gillespie_step(prey, predator, alpha, beta, gamma):
    '''Perform one step of the Gillespie simulation for a predator-prey system.
    Input:
    prey: current population of prey, integer
    predator: current population of predators, integer
    alpha: prey birth rate, float
    beta: predation rate, float
    gamma: predator death rate, float
    Output:
    time_step: time duration until next event occurs, a float; None if no event occurs
    prey: updated population of prey, integer
    predator: updated population of predators, integer
    event: a string describing the event that occurs ("prey_birth", "predation", or "predator_death"); None if no event occurs
    '''

    # Define a list of tuples for event names, rates, and their population changes
    event_list = [
        ("prey_birth", alpha * prey, lambda x, y: (x + 1, y)),
        ("predation", beta * prey * predator, lambda x, y: (x - 1, y + 1)),
        ("predator_death", gamma * predator, lambda x, y: (x, y - 1))
    ]

    # Calculate total rate
    total_rate = sum(rate for _, rate, _ in event_list)

    # If no events can occur, return early
    if total_rate == 0:
        return None, prey, predator, None

    # Sample the time until the next event
    time_step = np.random.exponential(1 / total_rate)

    # Select an event using a roulette wheel selection method
    r = np.random.uniform(0, total_rate)
    cumulative_rate = 0

    for event_name, rate, effect_func in event_list:
        cumulative_rate += rate
        if r <= cumulative_rate:
            prey, predator = effect_func(prey, predator)
            return time_step, prey, predator, event_name

    # Fallback in case no event is chosen (should not happen)
    return None, prey, predator, None



def evolve_LV(prey, predator, alpha, beta, gamma, T):
    '''Simulate the predator-prey dynamics using a Monte Carlo simulation with a focus on population stability.
    This function tracks and records the populations of prey and predators and the times at which changes occur.
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
    '''
    
    # Initialize lists to store simulation results
    time_cor = [0.0]
    prey_evol = [prey]
    predator_evol = [predator]
    
    current_time = 0.0
    dt = 0.05  # Define a small time step for consistent updates

    while current_time < T:
        # Calculate expected changes in prey and predator populations
        expected_prey_change = alpha * prey - beta * prey * predator
        expected_predator_change = beta * prey * predator - gamma * predator
        
        # Introduce variability using a Monte Carlo approach
        actual_prey_change = np.random.normal(expected_prey_change * dt, np.abs(expected_prey_change * dt))
        actual_predator_change = np.random.normal(expected_predator_change * dt, np.abs(expected_predator_change * dt))
        
        # Update populations
        prey = max(prey + actual_prey_change, 0)
        predator = max(predator + actual_predator_change, 0)
        
        # Update current time
        current_time += dt
        
        # Record current state
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)
        
        # Check for extinction events
        if prey <= 0 and predator <= 0:
            eco_event = "mutual extinction"
            return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event
        elif predator <= 0:
            eco_event = "predator extinction"
            return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event

    # Determine the ecological event if the simulation completed
    if predator > 0 and prey > 0:
        eco_event = "coexistence"
    elif predator <= 0:
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

    # Linearly interpolate the population data to a uniform time grid
    uniform_t = np.linspace(t[0], t[-1], num=1000)
    uniform_population = np.interp(uniform_t, t, population)

    # Calculate the autocorrelation function (ACF) of the population data
    autocorr = acf(uniform_population, fft=True, nlags=len(uniform_population)//2)

    # Identify the first significant peak in the autocorrelation function
    peak_indices = (np.diff(np.sign(np.diff(autocorr))) < 0).nonzero()[0] + 1
    significant_peaks = peak_indices[autocorr[peak_indices] > 0.1 * np.max(autocorr)]

    if len(significant_peaks) < 1:
        return float('nan')  # Return NaN if no significant peak is found

    # Calculate the time difference to the first significant peak
    peak_time = uniform_t[significant_peaks[0]]

    # Round periodicity to one decimal point
    periodicity = round(peak_time, 1)

    return periodicity



def predator_prey(prey, predator, alpha, beta, gamma, T):



    def gillespie(prey, predator, alpha, beta, gamma):
        events = [
            ("birth", alpha * prey, lambda p, q: (p + 1, q)),
            ("predation", beta * prey * predator, lambda p, q: (p - 1, q + 1)),
            ("death", gamma * predator, lambda p, q: (p, q - 1))
        ]
        total_rate = sum(rate for _, rate, _ in events)
        if total_rate == 0:
            return None, prey, predator
        delta_t = np.random.exponential(1 / total_rate)
        threshold = np.random.uniform(0, total_rate)
        cumulative_rate = 0
        for _, rate, update in events:
            cumulative_rate += rate
            if threshold < cumulative_rate:
                prey, predator = update(prey, predator)
                return delta_t, prey, predator
        return None, prey, predator

    def compute_period(time_series, population_series):
        if len(time_series) < 3:
            return 0.0
        interp_times = np.linspace(time_series[0], time_series[-1], 1000)
        interp_population = np.interp(interp_times, time_series, population_series)
        fft_vals = rfft(interp_population)
        fft_freqs = rfftfreq(len(interp_times), (interp_times[1] - interp_times[0]))
        dominant_frequency_index = np.argmax(np.abs(fft_vals))
        if fft_freqs[dominant_frequency_index] == 0:
            return 0.0
        period = 1.0 / fft_freqs[dominant_frequency_index]
        return round(period, 1)

    times = [0.0]
    prey_history = [prey]
    predator_history = [predator]
    current_time = 0.0

    while current_time < T:
        delta_t, prey, predator = gillespie(prey, predator, alpha, beta, gamma)
        if delta_t is None:
            break
        current_time += delta_t
        times.append(current_time)
        prey_history.append(prey)
        predator_history.append(predator)

        if prey <= 0 and predator <= 0:
            eco_event = "mutual extinction"
            return np.array(times), np.array(prey_history), np.array(predator_history), eco_event, 0.0, 0.0
        elif predator <= 0:
            eco_event = "predator extinction"
            return np.array(times), np.array(prey_history), np.array(predator_history), eco_event, 0.0, 0.0

    if predator > 0 and prey > 0:
        eco_event = "coexistence"
        prey_period = compute_period(times, prey_history)
        predator_period = compute_period(times, predator_history)
    else:
        eco_event = "mutual extinction" if prey <= 0 else "predator extinction"
        prey_period = predator_period = 0.0

    return np.array(times), np.array(prey_history), np.array(predator_history), eco_event, prey_period, predator_period


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