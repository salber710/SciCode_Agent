from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import random
from collections import Counter

def load_motif_from_df(data):
    '''Input:
    PWM matrix with keys 'A', 'C', 'G', 'T'
    Output:
    mat: (number of row of PWM matrix, 4) integer array, each row is a probability distribution
    '''


    # Initialize a matrix with shape (number of rows, 4) for storing smoothed values
    num_rows = len(data['A'])
    mat = np.zeros((num_rows, 4), dtype=float)

    # Define a function to smooth and normalize a row
    def smooth_and_normalize(a, c, g, t):
        values = np.array([a + 1, c + 1, g + 1, t + 1], dtype=float)
        return values / np.sum(values)

    # Iterate over each row index and apply smoothing and normalization
    for i in range(num_rows):
        mat[i] = smooth_and_normalize(data['A'][i], data['C'][i], data['G'][i], data['T'][i])

    return mat


def compute_kld(matrix):
    '''Input:
    (number of row of PWM matrix, 4) array, PWM
    Output:
    Kullback-Leibler divergence (float)
    '''
    # Define the uniform probability for A, T, C, G
    uniform_prob = 0.25

    # Initialize total KL divergence
    total_kld = 0.0

    # Iterate over each row in the PWM
    for row in matrix:
        # Compute KL divergence for the current row using a different approach
        kld_row = 0.0
        for prob in row:
            if prob > 0:
                # Use a custom log function based on Taylor series expansion
                # log2(x) ≈ (x - 1) - (x - 1)^2/2 + (x - 1)^3/3 ...
                x = prob / uniform_prob
                taylor_log2 = (x - 1) - (x - 1)**2 / 2 + (x - 1)**3 / 3
                kld_row += prob * taylor_log2

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




def scan_sequence(sequence, matrix, scale, num_runs=100):
    '''Input:
    DNA sequence (str)
    matrix (PWM)
    scale (float), 0<scale<1, 0.8 should be good, too low might cause false positive
    number of runs (int, default = 100)
    Output:
    Detected positions (int)
    '''

    sequence_length = len(sequence)
    motif_length = len(matrix)
    background_prob = 0.25

    # Calculate the threshold using expected value
    expected_value = motif_length * np.log2(background_prob)
    threshold = scale * expected_value

    # Nucleotide to index mapping
    nucleotide_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Function to compute log odds score for a given window
    def calculate_log_odds(window):
        return sum(np.log2(matrix[i][nucleotide_index[nuc]] / background_prob)
                   for i, nuc in enumerate(window) if nuc in nucleotide_index)

    # Use a numpy array to store detected positions for performance
    detected_positions = np.zeros(sequence_length, dtype=int)

    # Scanning process with random permutation of positions
    for _ in range(num_runs):
        for start in np.random.permutation(sequence_length - motif_length + 1):
            window = sequence[start:start + motif_length]
            log_odds_score = calculate_log_odds(window)
            if log_odds_score > threshold:
                detected_positions[start] += 1

    # Determine the most common position
    if np.any(detected_positions):
        most_common_position = np.argmax(detected_positions)
    else:
        most_common_position = None

    return most_common_position


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e