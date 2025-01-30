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
    event: a string describing the event that occurrs ("prey_birth", "predation", or "predator_death"); None if no event occurs
    '''
    
    # Calculate the rates of each possible event
    prey_birth_rate = alpha * prey
    predation_rate = beta * prey * predator
    predator_death_rate = gamma * predator
    
    # Total rate of all events
    total_rate = prey_birth_rate + predation_rate + predator_death_rate
    
    if total_rate == 0:
        return None, prey, predator, None
    
    # Sample the time until the next event from an exponential distribution
    time_step = np.random.exponential(1 / total_rate)
    
    # Determine which event occurs
    event_probabilities = [prey_birth_rate, predation_rate, predator_death_rate]
    cumulative_probabilities = np.cumsum(event_probabilities) / total_rate
    random_value = np.random.rand()
    
    # Determine the event
    if random_value < cumulative_probabilities[0]:
        # Prey birth event
        prey += 1
        event = "prey_birth"
    elif random_value < cumulative_probabilities[1]:
        # Predation event
        prey -= 1
        predator += 1
        event = "predation"
    else:
        # Predator death event
        predator -= 1
        event = "predator_death"
    
    return time_step, prey, predator, event


def evolve_LV(prey, predator, alpha, beta, gamma, T):
    '''Simulate the predator-prey dynamics using the Gillespie simulation algorithm.
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

    time_cor = [0]
    prey_evol = [prey]
    predator_evol = [predator]
    
    current_time = 0

    while current_time < T and prey > 0 and predator > 0:
        time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)
        
        if time_step is None:
            break
        
        current_time += time_step
        if current_time > T:
            break
        
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)

    # Determine the ecological event
    if prey > 0 and predator > 0:
        eco_event = "coexistence"
    elif prey == 0 and predator > 0:
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
    
    # Interpolate the population data to make it evenly spaced
    # Define an even time grid based on the min and max of t
    t_even = np.linspace(t.min(), t.max(), len(t))
    
    # Interpolate the population data onto this even time grid
    interp_func = interp1d(t, population, kind='cubic', fill_value='extrapolate')
    population_even = interp_func(t_even)
    
    # Perform the Fourier Transform on the evenly spaced data
    population_fft = fft(population_even)
    
    # Get the frequencies corresponding to the FFT components
    frequencies = fftfreq(len(t_even), (t_even[1] - t_even[0]))
    
    # Get the magnitudes of the FFT components
    magnitudes = np.abs(population_fft)
    
    # Find the frequency with the maximum magnitude (ignoring the zero frequency)
    dominant_frequency = frequencies[np.argmax(magnitudes[1:]) + 1]
    
    # Calculate the period from the dominant frequency
    if dominant_frequency != 0:
        periodicity = 1 / dominant_frequency
    else:
        periodicity = np.inf  # In case no valid frequency is found
    
    # Round the period to one decimal point
    periodicity = round(periodicity, 1)
    
    return periodicity


try:
    targets = process_hdf5_to_tuple('53.3', 3)
    target = targets[0]
    time_cor = np.linspace(0., 5., 500)
    prey_evol = np.sin(5.*time_cor) + 2.
    assert np.allclose(spectral_periodicity(time_cor, prey_evol), target)

    target = targets[1]
    time_cor = np.linspace(0., 10., 500)
    prey_evol = np.cos(10.*time_cor) + 2.
    assert np.allclose(spectral_periodicity(time_cor, prey_evol), target)

    target = targets[2]
    time_cor = np.linspace(0., 20., 500)
    prey_evol = np.sin(10*time_cor) + np.cos(10*time_cor) + 5.
    assert np.allclose(spectral_periodicity(time_cor, prey_evol), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e