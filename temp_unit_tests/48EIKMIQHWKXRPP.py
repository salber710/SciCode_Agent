
import unittest
import numpy as np
from WFSRK8DJQKY6QHR import blahut_arimoto

class TestBlahutArimoto(unittest.TestCase):
    def test_uniform_channel(self):
        channel = np.array([[0.5, 0.5], [0.5, 0.5]])
        e = 0.01
        expected_capacity = 1.0
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=2)

    def test_zero_error_threshold(self):
        channel = np.array([[0.7, 0.3], [0.3, 0.7]])
        e = 0.0
        expected_capacity = 0.8813
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=4)

    def test_small_matrix(self):
        channel = np.array([[1.0]])
        e = 0.01
        expected_capacity = 0.0
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=2)

    def test_large_uniform_channel(self):
        channel = np.full((10, 10), 0.1)
        e = 0.01
        expected_capacity = np.log2(10)
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=2)

    def test_channel_with_zeros(self):
        channel = np.array([[0.8, 0.2, 0.0], [0.0, 0.5, 0.5], [0.2, 0.3, 0.5]])
        e = 0.01
        expected_capacity = 1.5709
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=4)

    def test_non_square_channel(self):
        channel = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        e = 0.01
        expected_capacity = 0.9709
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=4)

    def test_high_precision(self):
        channel = np.array([[0.9, 0.1], [0.1, 0.9]])
        e = 1e-6
        expected_capacity = 0.4690
        self.assertAlmostEqual(blahut_arimoto(channel, e), expected_capacity, places=4)

    def test_invalid_channel(self):
        channel = np.array([[1.1, -0.1], [0.0, 1.0]])
        e = 0.01
        with self.assertRaises(ValueError):
            blahut_arimoto(channel, e)

if __name__ == '__main__':
    unittest.main()
