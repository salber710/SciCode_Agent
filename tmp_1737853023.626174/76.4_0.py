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
    
    # Check if the input is empty
    if pwm_array.size == 0:
        return np.array([]).reshape(0, 4)
    
    # Check for negative numbers and raise ValueError if found
    if np.any(pwm_array < 0):
        raise ValueError("Negative values are not allowed in PWM matrix.")
    
    # Check for consistent lengths in the input data
    if len(set(map(len, [data['A'], data['C'], data['G'], data['T']])) ) != 1:
        raise ValueError("All input lists must have the same length.")
    
    # Add 1 to each entry to avoid log divergence
    pwm_array = pwm_array + 1
    
    # Normalize each row using L1 normalization
    row_sums = pwm_array.sum(axis=1, keepdims=True)
    mat = pwm_array / row_sums
    
    return mat


# Background: The Kullback-Leibler (KL) divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. In the context of a Position Weight Matrix (PWM), we want to measure how the observed distribution of nucleotides at each position (given by the PWM) diverges from a uniform background distribution. The uniform distribution assumes that each nucleotide (A, C, G, T) appears with equal probability, i.e., 0.25. The KL divergence for a single position is calculated as the sum over all nucleotides of the observed probability times the log of the observed probability divided by the background probability. The total KL divergence for the PWM is the sum of the KL divergences for each position.


def compute_kld(matrix):
    '''Input:
    (number of row of PWM matrix, 4) array, PWM
    Output:
    Kullback-Leibler divergence (float)
    '''
    # Define the uniform background distribution
    background_prob = 0.25
    
    # Initialize the total KL divergence
    total_kld = 0.0
    
    # Check for invalid input
    if matrix.size == 0:
        raise ValueError("Input matrix is empty.")
    if matrix.shape[1] != 4:
        raise ValueError("Input matrix must have exactly 4 columns.")
    
    # Check for negative values
    if np.any(matrix < 0):
        raise ValueError("Input matrix must not contain negative values.")
    
    # Check for invalid probability distributions
    if not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("Each row of the input matrix must sum to 1.")
    
    # Check for zero values and compute KL divergence
    for row in matrix:
        if np.any(row == 0):
            raise ValueError("Input matrix must not contain zero values, as log(0) is undefined.")
        kld_row = np.sum(row * np.log(row / background_prob))
        total_kld += kld_row
    
    return total_kld

def generate_dna(N: int, PWM: dict) -> tuple:
    """
    Input:
    N (int): Length of the resultant DNA sequence.
    PWM matrix with keys 'A', 'C', 'G', 'T'

    Output:
    tuple: Insertion location (int), DNA sequence (str), DNA reverse complement (str)
    """
    p = random.randint(0, N - 1)
    nucleotide = 'ACGT'
    uni_weights = [0.25, 0.25, 0.25, 0.25]
    dna_string = ''.join(random.choices(nucleotide, uni_weights, k=N))
    spike_mat = load_motif_from_df(PWM)
    spiked_seq = ''.join((random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0] for i in range(len(PWM['A']))))
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_seq = dna_string[::-1]
    reverse_complement = ''.join((complement[nuc] for nuc in reversed_seq if nuc in complement))
    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[:N - p] + spiked_seq + reverse_complement[N - p:]
    return (p, new_seq, new_seq_rc)


# Background: In bioinformatics, scanning a DNA sequence with a Position Weight Matrix (PWM) involves sliding a window
# across the sequence and calculating the log-odds score for each window position. The log-odds score is a measure of
# how much more likely the observed sequence is under the PWM model compared to a background model, which is assumed
# to be uniform in this case. The log-odds score for a window is computed by summing the log of the ratio of the PWM
# probability to the background probability (0.25 for each nucleotide) for each nucleotide in the window. If the score
# exceeds a threshold, which is the product of a scale factor and the expected value of the PWM, the position is recorded.
# By running this scanning process multiple times, we can identify the most frequently detected positions, which may
# indicate significant motif occurrences.




def scan_sequence(sequence, matrix, scale, num_runs=100):
    '''Input:
    DNA sequence (str)
    matrix (PWM)
    scale (float) 0<scale<1 , 0.8 should be good, too low might cause false positive
    number of run (int, default = 100)
    Output:
    Detected positions (int)
    '''
    # Define the uniform background probability
    background_prob = 0.25
    
    # Calculate the expected value of the PWM
    expected_value = np.sum(matrix * np.log(matrix / background_prob))
    
    # Calculate the threshold
    threshold = scale * expected_value
    
    # Length of the PWM motif
    motif_length = matrix.shape[0]
    
    # List to store detected positions
    detected_positions = []
    
    # Run the scanner multiple times
    for _ in range(num_runs):
        # Scan through the sequence
        for i in range(len(sequence) - motif_length + 1):
            # Extract the current window
            window = sequence[i:i + motif_length]
            
            # Calculate the log-odds score for the current window
            log_odds_score = 0.0
            for j, nucleotide in enumerate(window):
                if nucleotide in 'ACGT':
                    index = 'ACGT'.index(nucleotide)
                    log_odds_score += np.log(matrix[j, index] / background_prob)
            
            # Check if the log-odds score exceeds the threshold
            if log_odds_score > threshold:
                detected_positions.append(i)
    
    # Find the most common position
    if detected_positions:
        most_common_position = Counter(detected_positions).most_common(1)[0][0]
    else:
        most_common_position = None
    
    return most_common_position

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('76.4', 6)
target = targets[0]

random.seed(42)
data = {
    'A': [12.50, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00, 55],
    'C': [33.33, 0.00, 95.83, 0.00, 0.00, 0.00, 0.00, 35],
    'G': [8.33, 95.83, 0.00, 95.83, 0.00, 100.00, 0.00, 10],
    'T': [45.83, 4.17, 4.17, 4.17, 100.00, 0.00, 0.00, 0.00],
}
inserted_position, sequence, sequence_rc = generate_dna(240,data)
assert np.allclose(scan_sequence(sequence, load_motif_from_df(data),0.8), target)
target = targets[1]

random.seed(42)
data2 = {
    'A': [12.50, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00, 100,50,60],
    'C': [33.33, 0.00, 95.83, 0.00, 0.00, 0.00, 0.00, 0,50,40],
    'G': [8.33, 95.83, 0.00, 95.83, 0.00, 100.00, 0.00, 0,0,0],
    'T': [45.83, 4.17, 4.17, 4.17, 100.00, 0.00, 0.00, 0.00,0,0],
}
inserted_position, sequence, sequence_rc = generate_dna(1000,data2)
assert np.allclose(scan_sequence(sequence, load_motif_from_df(data2),0.8), target)
target = targets[2]

random.seed(42)
data3 = {
    'A': [25.0,25.0,25.0,25.0,25.0,25.0],
    'C': [25.0,25.0,25.0,25.0,25.0,25.0],
    'G': [25.0,25.0,25.0,25.0,25.0,25.0],
    'T': [25.0,25.0,25.0,25.0,25.0,25.0],
}
inserted_position, sequence, sequence_rc = generate_dna(1000,data3)
assert scan_sequence(sequence, load_motif_from_df(data3),0.8) == None
target = targets[3]

random.seed(42)
data = {
    'A': [12.50, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00, 55],
    'C': [33.33, 0.00, 95.83, 0.00, 0.00, 0.00, 0.00, 35],
    'G': [8.33, 95.83, 0.00, 95.83, 0.00, 100.00, 0.00, 10],
    'T': [45.83, 4.17, 4.17, 4.17, 100.00, 0.00, 0.00, 0.00],
}
inserted_position, sequence, sequence_rc = generate_dna(240,data)
assert np.allclose(scan_sequence(sequence, load_motif_from_df(data),0.8), target)
target = targets[4]

random.seed(42)
data2 = {
    'A': [12.50, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00, 100,50,60],
    'C': [33.33, 0.00, 95.83, 0.00, 0.00, 0.00, 0.00, 0,50,40],
    'G': [8.33, 95.83, 0.00, 95.83, 0.00, 100.00, 0.00, 0,0,0],
    'T': [45.83, 4.17, 4.17, 4.17, 100.00, 0.00, 0.00, 0.00,0,0],
}
inserted_position, sequence, sequence_rc = generate_dna(1000,data2)
assert np.allclose(scan_sequence(sequence, load_motif_from_df(data),0.8), target)
target = targets[5]

# Since the data3 we are using here is exactly uniform distribution
# This case should not detect any position (Scanner Output should be None)
random.seed(42)
data3 = {
    'A': [25.0,25.0,25.0,25.0,25.0,25.0],
    'C': [25.0,25.0,25.0,25.0,25.0,25.0],
    'G': [25.0,25.0,25.0,25.0,25.0,25.0],
    'T': [25.0,25.0,25.0,25.0,25.0,25.0],
}
inserted_position, sequence, sequence_rc = generate_dna(1000,data3)
assert scan_sequence(sequence, load_motif_from_df(data),0.8) == None
