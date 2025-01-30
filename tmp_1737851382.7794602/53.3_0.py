import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fft, fftfreq

# Background: The Lotka-Volterra equations describe the dynamics of a biological system in which two species interact, a predator and its prey. 
# The equations are a pair of first-order, non-linear, differential equations. The Gillespie algorithm is a stochastic simulation method used 
# to simulate the time evolution of a system with discrete events. In this context, the events are "prey_birth", "predation", and "predator_death". 
# The algorithm involves calculating the rates of these events, sampling the time to the next event from an exponential distribution, and 
# updating the system state based on which event occurs. The rate of "prey_birth" is proportional to the prey population and the birth rate alpha. 
# The rate of "predation" is proportional to the product of prey and predator populations and the predation rate beta. 
# The rate of "predator_death" is proportional to the predator population and the death rate gamma.


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

    # Ensure non-negative initial populations
    prey = max(prey, 0)
    predator = max(predator, 0)

    # Calculate the rates of each event
    rate_prey_birth = max(alpha * prey, 0)
    rate_predation = max(beta * prey * predator, 0)
    rate_predator_death = max(gamma * predator, 0)

    # Total rate of all events
    total_rate = rate_prey_birth + rate_predation + rate_predator_death

    # If no events can occur, return None
    if total_rate == 0:
        return None, prey, predator, None

    # Sample the time to the next event from an exponential distribution
    time_step = np.random.exponential(1 / total_rate)

    # Determine which event occurs
    event_probabilities = np.array([rate_prey_birth, rate_predation, rate_predator_death]) / total_rate
    event = np.random.choice(["prey_birth", "predation", "predator_death"], p=event_probabilities)

    # Update populations based on the event
    if event == "prey_birth":
        prey += 1
    elif event == "predation":
        prey -= 1
        predator += 1
    elif event == "predator_death":
        predator -= 1

    return time_step, prey, predator, event


# Background: The Lotka-Volterra equations model the interaction between predator and prey populations. The Gillespie algorithm is a stochastic
# simulation method that allows us to simulate the time evolution of these populations by considering discrete events such as prey birth, predation,
# and predator death. In this simulation, we track the populations over time until a specified final time T. We also identify ecological events:
# "coexistence" occurs if both populations are non-zero at the end of the simulation, "predator extinction" if predators reach zero while prey
# remain, and "mutual extinction" if both populations reach zero.


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

    if prey < 0 or predator < 0:
        raise ValueError("Initial populations of prey and predators must be non-negative.")

    # Initialize time and record initial conditions
    current_time = 0.0
    time_cor = [current_time]
    prey_evol = [prey]
    predator_evol = [predator]

    while current_time < T:
        # Perform a Gillespie step
        time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)

        # If no event occurs, end the simulation
        if event is None:
            break

        # Update the current time
        current_time += time_step

        # Record the new state
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)

        # Check if the simulation should end early due to extinction
        if prey == 0 and predator == 0:
            break
        if predator == 0:
            break

    # Determine the ecological event
    if prey > 0 and predator > 0:
        eco_event = "coexistence"
    elif predator == 0 and prey > 0:
        eco_event = "predator extinction"
    else:
        eco_event = "mutual extinction"

    return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event

def gillespie_step(prey, predator, alpha, beta, gamma):
    ''' Perform one step of the Gillespie simulation algorithm.
    Input:
    prey: current population of prey
    predator: current population of predators
    alpha: prey birth rate
    beta: predation rate
    gamma: predator death rate
    Output:
    time_step: time until the next event
    prey: updated population of prey
    predator: updated population of predators
    event: type of event that occurred ('birth', 'death', or 'predation')
    '''
    if prey == 0 and predator == 0:
        return 0, 0, 0, None

    # Calculate rates
    rate_birth = alpha * prey
    rate_predation = beta * prey * predator
    rate_death = gamma * predator

    # Total rate of all events
    rate_total = rate_birth + rate_predation + rate_death

    if rate_total == 0:
        return 0, prey, predator, None

    # Time until next event
    time_step = np.random.exponential(1 / rate_total)

    # Determine which event occurs
    rand = np.random.uniform(0, rate_total)
    if rand < rate_birth:
        prey += 1
        event = 'birth'
    elif rand < rate_birth + rate_predation:
        prey -= 1
        predator += 1
        event = 'predation'
    else:
        predator -= 1
        event = 'death'

    return time_step, prey, predator, event



# Background: To determine the periodicity of a population's oscillation in a predator-prey system, we can use spectral analysis.
# The Fourier Transform is a mathematical technique that transforms a time-domain signal into its constituent frequencies.
# By analyzing the frequency spectrum of the population data, we can identify the dominant frequency, which corresponds to the periodicity of the oscillation.
# Given that the time steps are uneven, we first need to interpolate the data to a uniform time grid before applying the Fourier Transform.
# The dominant frequency can then be used to calculate the period of oscillation, which is the inverse of the frequency.




def spectral_periodicity(t, population):
    '''Estimate the periodicity of population with uneven time step and stochasticity.
    Input:
    t: time coordinates of population evolution, 1D array of floats
    population: evolution history of population of some species, 1D array of floats (same size as t)
    Output:
    periodicity: estimated periodicity, float rounded up to one decimal point.
    '''

    # Interpolate the population data to a uniform time grid
    uniform_time = np.linspace(t[0], t[-1], num=len(t))
    interp_func = interp1d(t, population, kind='linear')
    uniform_population = interp_func(uniform_time)

    # Perform Fourier Transform to find frequency components
    population_fft = fft(uniform_population)
    frequencies = fftfreq(len(uniform_population), d=(uniform_time[1] - uniform_time[0]))

    # Find the dominant frequency (ignoring the zero frequency)
    positive_frequencies = frequencies[frequencies > 0]
    positive_magnitudes = np.abs(population_fft[frequencies > 0])
    dominant_frequency = positive_frequencies[np.argmax(positive_magnitudes)]

    # Calculate the period from the dominant frequency
    periodicity = 1 / dominant_frequency

    # Round the periodicity to one decimal point
    return round(periodicity, 1)

from scicode.parse.parse import process_hdf5_to_tuple
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
