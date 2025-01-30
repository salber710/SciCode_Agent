import numpy as np
import random
from collections import Counter



# Background: A Position Weight Matrix (PWM) is a common representation of motifs in biological sequences, 
# particularly in DNA sequences. Each row in a PWM corresponds to a position in the motif, and each column 
# corresponds to a nucleotide (A, C, G, T). The values in the PWM represent the frequency or weight of each 
# nucleotide at each position. To convert these weights into a probability distribution, we can normalize 
# each row using L1 normalization. L1 normalization involves dividing each element by the sum of the elements 
# in the row, ensuring that the sum of the row equals 1. Adding 1 to each entry before normalization is a 
# technique to avoid issues with log divergence, especially when dealing with zero values.


def load_motif_from_df(data):
    '''Input:
    PWM matrix with keys 'A', 'C', 'G', 'T'
    Output:
    mat: (number of row of PWM matrix, 4) integer array, each row is a probability distribution
    '''
    # Convert the input dictionary to a numpy array
    pwm_array = np.array([data['A'], data['C'], data['G'], data['T']]).T
    
    # Add 1 to each entry to avoid log divergence
    pwm_array += 1
    
    # Normalize each row using L1 normalization
    row_sums = pwm_array.sum(axis=1, keepdims=True)
    mat = pwm_array / row_sums
    
    return mat

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('76.1', 3)
target = targets[0]

data = {
    'A': [12.50, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00, 55],
    'C': [33.33, 0.00, 95.83, 0.00, 0.00, 0.00, 0.00, 35],
    'G': [8.33, 95.83, 0.00, 95.83, 0.00, 100.00, 0.00, 10],
    'T': [45.83, 4.17, 4.17, 4.17, 100.00, 0.00, 0.00, 0.00],
}
assert np.allclose(load_motif_from_df(data), target)
target = targets[1]

data2 = {
    'A': [12.50, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00, 100,50,60],
    'C': [33.33, 0.00, 95.83, 0.00, 0.00, 0.00, 0.00, 0,50,40],
    'G': [8.33, 95.83, 0.00, 95.83, 0.00, 100.00, 0.00, 0,0,0],
    'T': [45.83, 4.17, 4.17, 4.17, 100.00, 0.00, 0.00, 0.00,0,0],
}
assert np.allclose(load_motif_from_df(data2), target)
target = targets[2]

data3 = {
    'A': [25.0,25.0,25.0,25.0,25.0,25.0],
    'C': [25.0,25.0,25.0,25.0,25.0,25.0],
    'G': [25.0,25.0,25.0,25.0,25.0,25.0],
    'T': [25.0,25.0,25.0,25.0,25.0,25.0],
}
assert np.allclose(load_motif_from_df(data3), target)
