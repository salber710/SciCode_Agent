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
    
    # Initialize variables to record the evolution
    time_cor = [0]
    prey_evol = [prey]
    predator_evol = [predator]
    
    current_time = 0
    
    while current_time < T:
        # Perform one step of the Gillespie simulation
        time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)
        
        # If no event occurs, break the loop
        if event is None:
            break
        
        # Update the current time
        current_time += time_step
        
        # Record the time and population changes
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)
    
    # Determine the ecological event
    if prey > 0 and predator > 0:
        eco_event = "coexistence"
    elif predator == 0 and prey > 0:
        eco_event = "predator extinction"
    elif prey == 0:
        eco_event = "mutual extinction"
    else:
        eco_event = "coexistence"  # default case if both populations remain positive
    
    return np.array(time_cor), np.array(prey_evol), np.array(predator_evol), eco_event


try:
    targets = process_hdf5_to_tuple('53.2', 3)
    target = targets[0]
    prey, predator = 200, 200
    alpha, beta, gamma = 2., 0.01, 3.
    T = 2.
    np.random.seed(2)
    time_cor, prey_evol, predator_evol, eco_event = evolve_LV(prey, predator, alpha, beta, gamma, T)
    a, b, c, d = target
    assert np.allclose(time_cor, a) and np.allclose(prey_evol, b) and np.allclose(predator_evol, c) and eco_event == d

    target = targets[1]
    prey, predator = 100, 20
    alpha, beta, gamma = 3., 0.05, 1.
    T = 8.
    np.random.seed(1)
    time_cor, prey_evol, predator_evol, eco_event = evolve_LV(prey, predator, alpha, beta, gamma, T)
    a, b, c, d = target
    assert np.allclose(time_cor, a) and np.allclose(prey_evol, b) and np.allclose(predator_evol, c) and eco_event == d

    target = targets[2]
    prey, predator = 100, 10
    alpha, beta, gamma = 1., 0.005, 5.
    T = 2.
    np.random.seed(3)
    time_cor, prey_evol, predator_evol, eco_event = evolve_LV(prey, predator, alpha, beta, gamma, T)
    a, b, c, d = target
    assert np.allclose(time_cor, a) and np.allclose(prey_evol, b) and np.allclose(predator_evol, c) and eco_event == d

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e