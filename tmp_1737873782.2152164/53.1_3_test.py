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
    
    # Calculate rates of each event
    prey_birth_rate = alpha * prey
    predation_rate = beta * prey * predator
    predator_death_rate = gamma * predator
    
    # Total rate of all events
    total_rate = prey_birth_rate + predation_rate + predator_death_rate
    
    if total_rate == 0:
        return None, prey, predator, None
    
    # Sample time step from exponential distribution with the total rate
    time_step = np.random.exponential(1 / total_rate)
    
    # Determine which event occurs
    event_probabilities = np.array([prey_birth_rate, predation_rate, predator_death_rate]) / total_rate
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


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e