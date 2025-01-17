import numpy as np
from scipy.interpolate import interp1d
from numpy.fft import fft, fftfreq

# Background: The Lotka-Volterra equations describe the dynamics of biological systems in which two species interact, predator and prey. 
# The equations are a pair of first-order, non-linear, differential equations. The Gillespie algorithm is a stochastic simulation 
# method used to simulate the time evolution of a system with discrete events. In this context, it is used to simulate the 
# predator-prey interactions by randomly determining the time until the next event (prey birth, predation, or predator death) 
# and which event occurs. The time until the next event is sampled from an exponential distribution, where the rate is determined 
# by the sum of the rates of all possible events. The event that occurs is chosen based on the relative probabilities of each event.


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

    # Calculate the rates of each event
    rate_prey_birth = alpha * prey
    rate_predation = beta * prey * predator
    rate_predator_death = gamma * predator

    # Total rate of all events
    total_rate = rate_prey_birth + rate_predation + rate_predator_death

    # If no events can occur, return None
    if total_rate == 0:
        return None, prey, predator, None

    # Sample the time until the next event from an exponential distribution
    time_step = np.random.exponential(1 / total_rate)

    # Determine which event occurs
    event_probabilities = np.array([rate_prey_birth, rate_predation, rate_predator_death]) / total_rate
    event_choice = np.random.choice(['prey_birth', 'predation', 'predator_death'], p=event_probabilities)

    # Update populations based on the event
    if event_choice == 'prey_birth':
        prey += 1
    elif event_choice == 'predation':
        prey -= 1
        predator += 1
    elif event_choice == 'predator_death':
        predator -= 1

    return time_step, prey, predator, event_choice


# Background: The Lotka-Volterra equations model the interaction between predator and prey populations. The Gillespie algorithm
# is a stochastic method that simulates these interactions by determining the time until the next event and which event occurs.
# In this simulation, we track the evolution of prey and predator populations over time until a specified final time T.
# The simulation records the time and population changes, and identifies ecological events such as "coexistence" (both populations
# survive), "predator extinction" (predators die out), or "mutual extinction" (both populations die out).

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


    # Initialize time and record arrays
    current_time = 0.0
    time_cor = [current_time]
    prey_evol = [prey]
    predator_evol = [predator]

    # Run the simulation until the specified final time T
    while current_time < T:
        # Perform a single Gillespie step
        time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)

        # If no event occurs, break the loop
        if event is None:
            break

        # Update the current time
        current_time += time_step

        # Record the time and populations
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)

        # Check for extinction events
        if prey == 0 and predator == 0:
            eco_event = "mutual extinction"
            return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event
        elif predator == 0:
            eco_event = "predator extinction"
            return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event

    # If the loop completes without extinction, the populations coexist
    eco_event = "coexistence"
    return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event


# Background: In order to determine the periodicity of a population's oscillation in a predator-prey system, we can use spectral analysis.
# The Fourier Transform is a mathematical technique that transforms a time-domain signal into its constituent frequencies, revealing
# the dominant frequencies of oscillation. Given the stochastic nature of the data and uneven time steps, we first need to interpolate
# the data to a uniform time grid. Then, we apply the Fast Fourier Transform (FFT) to identify the dominant frequency, which corresponds
# to the periodicity of the population oscillations. The periodicity is the inverse of the frequency and should be rounded to one decimal
# point for precision.




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

    # Perform FFT on the interpolated population data
    population_fft = fft(uniform_population)
    frequencies = fftfreq(len(uniform_time), d=(uniform_time[1] - uniform_time[0]))

    # Find the dominant frequency (ignoring the zero frequency)
    positive_frequencies = frequencies[frequencies > 0]
    positive_magnitudes = np.abs(population_fft[frequencies > 0])
    dominant_frequency = positive_frequencies[np.argmax(positive_magnitudes)]

    # Calculate the periodicity as the inverse of the dominant frequency
    periodicity = 1 / dominant_frequency

    # Round the periodicity to one decimal point
    return round(periodicity, 1)



# Background: The Lotka-Volterra equations model predator-prey interactions, and the Gillespie algorithm is a stochastic method
# for simulating these interactions. The algorithm determines the time until the next event and which event occurs, allowing us
# to simulate the evolution of prey and predator populations over time. By recording the population changes and time coordinates,
# we can identify ecological events such as "coexistence", "predator extinction", or "mutual extinction". If coexistence occurs,
# we can analyze the periodicity of population oscillations using spectral analysis. This involves interpolating the data to a
# uniform time grid and applying the Fast Fourier Transform (FFT) to find the dominant frequency, which indicates the periodicity
# of the oscillations.

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
    predator_period: estimated periodicity of redator population, float rounded up to one decimal point; 0.0 if no coexistence
    '''




    def gillespie_step(prey, predator, alpha, beta, gamma):
        rate_prey_birth = alpha * prey
        rate_predation = beta * prey * predator
        rate_predator_death = gamma * predator
        total_rate = rate_prey_birth + rate_predation + rate_predator_death

        if total_rate == 0:
            return None, prey, predator, None

        time_step = np.random.exponential(1 / total_rate)
        event_probabilities = np.array([rate_prey_birth, rate_predation, rate_predator_death]) / total_rate
        event_choice = np.random.choice(['prey_birth', 'predation', 'predator_death'], p=event_probabilities)

        if event_choice == 'prey_birth':
            prey += 1
        elif event_choice == 'predation':
            prey -= 1
            predator += 1
        elif event_choice == 'predator_death':
            predator -= 1

        return time_step, prey, predator, event_choice

    def spectral_periodicity(t, population):
        uniform_time = np.linspace(t[0], t[-1], num=len(t))
        interp_func = interp1d(t, population, kind='linear')
        uniform_population = interp_func(uniform_time)

        population_fft = fft(uniform_population)
        frequencies = fftfreq(len(uniform_time), d=(uniform_time[1] - uniform_time[0]))

        positive_frequencies = frequencies[frequencies > 0]
        positive_magnitudes = np.abs(population_fft[frequencies > 0])
        dominant_frequency = positive_frequencies[np.argmax(positive_magnitudes)]

        periodicity = 1 / dominant_frequency
        return round(periodicity, 1)

    current_time = 0.0
    time_cor = [current_time]
    prey_evol = [prey]
    predator_evol = [predator]

    while current_time < T:
        time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)

        if event is None:
            break

        current_time += time_step
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)

        if prey == 0 and predator == 0:
            eco_event = "mutual extinction"
            return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event, 0.0, 0.0
        elif predator == 0:
            eco_event = "predator extinction"
            return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event, 0.0, 0.0

    eco_event = "coexistence"
    prey_period = spectral_periodicity(time_cor, prey_evol)
    predator_period = spectral_periodicity(time_cor, predator_evol)

    return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event, prey_period, predator_period


from scicode.parse.parse import process_hdf5_to_tuple

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
