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



# Background: The Lotka-Volterra equations model the dynamics of biological systems in which two species interact, predator and prey. 
# The Gillespie algorithm is a stochastic simulation method used to simulate the time evolution of a system with discrete events. 
# In this context, it is used to simulate the predator-prey dynamics by randomly determining the time and type of the next event 
# (prey birth, predation, or predator death) based on the rates of these events. The simulation continues until a specified final 
# time T is reached or no further events can occur. The ecological events of interest are "coexistence" (both populations survive), 
# "predator extinction" (predators die out), and "mutual extinction" (both populations die out).

def evolve_LV(prey, predator, alpha, beta, gamma, T):

    
    # Initialize time and record arrays
    current_time = 0.0
    time_cor = [current_time]
    prey_evol = [prey]
    predator_evol = [predator]
    
    # Run the simulation until the final time T
    while current_time < T:
        # Perform a Gillespie step
        time_step, prey, predator, event = gillespie_step(prey, predator, alpha, beta, gamma)
        
        # If no event can occur, break the loop
        if event is None:
            break
        
        # Update the current time
        current_time += time_step
        
        # Record the time and populations
        time_cor.append(current_time)
        prey_evol.append(prey)
        predator_evol.append(predator)
    
    # Determine the ecological event
    if prey > 0 and predator > 0:
        eco_event = "coexistence"
    elif predator == 0:
        eco_event = "predator extinction"
    else:
        eco_event = "mutual extinction"
    
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