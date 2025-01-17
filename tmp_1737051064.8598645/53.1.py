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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('53.1', 3)
target = targets[0]

prey, predator = 200, 200
alpha, beta, gamma = 2., 0.01, 3.
np.random.seed(2)
time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)
a, b, c, d = target
assert np.allclose(time_step, a) and np.allclose(prey, b) and np.allclose(predator, c) and event == d
target = targets[1]

prey, predator = 100, 20
alpha, beta, gamma = 3., 0.05, 1.
np.random.seed(1)
time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)
a, b, c, d = target
assert np.allclose(time_step, a) and np.allclose(prey, b) and np.allclose(predator, c) and event == d
target = targets[2]

prey, predator = 100, 10
alpha, beta, gamma = 1., 0.005, 5.
np.random.seed(3)
time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)
a, b, c, d = target
assert np.allclose(time_step, a) and np.allclose(prey, b) and np.allclose(predator, c) and event == d
