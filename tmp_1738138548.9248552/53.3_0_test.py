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



# Background: In order to determine the periodicity of a population's oscillation in a predator-prey model, we can use spectral analysis. 
# The Fourier Transform is a mathematical technique that transforms a time-domain signal into its constituent frequencies. 
# By applying the Fast Fourier Transform (FFT) to the population data, we can identify the dominant frequency, which corresponds to the periodicity of the oscillation. 
# However, since the data may have uneven time steps, we first need to interpolate the data to a uniform time grid before applying FFT. 
# The frequency with the highest amplitude in the FFT result indicates the dominant periodicity.




def spectral_periodicity(t, population):
    '''Estimate the periodicity of population with uneven time step and stochasticity.
    Input:
    t: time coordinates of population evolution, 1D array of floats
    population: evolution history of population of some species, 1D array of floats (same size as t)
    Output:
    periodicity: estimated periodicity, float rounded up to one decimal point.
    '''
    
    # Interpolate the population data to a uniform time grid
    uniform_time = np.linspace(t[0], t[-1], len(t))
    interp_func = interp1d(t, population, kind='linear')
    uniform_population = interp_func(uniform_time)
    
    # Apply FFT to the uniformly sampled population data
    fft_result = fft(uniform_population)
    freqs = fftfreq(len(uniform_population), d=(uniform_time[1] - uniform_time[0]))
    
    # Find the frequency with the highest amplitude (ignoring the zero frequency)
    positive_freqs = freqs[freqs > 0]
    positive_amplitudes = np.abs(fft_result[freqs > 0])
    
    # Identify the dominant frequency
    dominant_freq_index = np.argmax(positive_amplitudes)
    dominant_freq = positive_freqs[dominant_freq_index]
    
    # Calculate the period from the dominant frequency
    periodicity = 1 / dominant_freq
    
    # Round the periodicity to one decimal point
    return round(periodicity, 1)


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