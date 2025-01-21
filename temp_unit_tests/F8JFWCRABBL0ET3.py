
import numpy as np
import unittest

def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    '''
    delta = np.array(r1) - np.array(r2)
    delta = delta - L * np.round(delta / L)
    return delta

class TestDistV(unittest.TestCase):
    def test_basic(self):
        r1 = np.array([1, 1, 1])
        r2 = np.array([2, 2, 2])
        L = 10
        expected = np.array([-1, -1, -1])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_edge_crossing(self):
        r1 = np.array([9, 9, 9])
        r2 = np.array([1, 1, 1])
        L = 10
        expected = np.array([-2, -2, -2])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_large_L(self):
        r1 = np.array([100, 100, 100])
        r2 = np.array([300, 300, 300])
        L = 500
        expected = np.array([-200, -200, -200])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_negative_coordinates(self):
        r1 = np.array([-5, -5, -5])
        r2 = np.array([-10, -10, -10])
        L = 20
        expected = np.array([5, 5, 5])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_zero_length(self):
        r1 = np.array([0, 0, 0])
        r2 = np.array([0, 0, 0])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_zero_box_size(self):
        r1 = np.array([1, 2, 3])
        r2 = np.array([4, 5, 6])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            dist_v(r1, r2, L)

    def test_identical_points(self):
        r1 = np.array([5, 5, 5])
        r2 = np.array([5, 5, 5])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_fractional_box_size(self):
        r1 = np.array([0.1, 0.1, 0.1])
        r2 = np.array([0.9, 0.9, 0.9])
        L = 1.0
        expected = np.array([-0.8, -0.8, -0.8])
        np.testing.assert_array_almost_equal(dist_v(r1, r2, L), expected)

    def test_within_half_box_length(self):
        r1 = [1, 1, 1]
        r2 = [2, 2, 2]
        L = 10
        expected = np.array([-1, -1, -1])
        np.testing.assert_array_equal(dist_v(r1, r2, L), expected)

    def test_exactly_half_box_length(self):
        r1 = [0, 0, 0]
        r2 = [5, 5, 5]
        L = 10
        expected = np.array([-5, -5, -5])
        np.testing.assert_array_equal(dist_v(r1, r2, L), expected)

    def test_greater_than_half_box_length(self):
        r1 = [1, 1, 1]
        r2 = [9, 9, 9]
        L = 10
        expected = np.array([2, 2, 2])
        np.testing.assert_array_equal(dist_v(r1, r2, L), expected)

    def test_large_coordinates(self):
        r1 = [100, 100, 100]
        r2 = [105, 105, 105]
        L = 10
        expected = np.array([-5, -5, -5])
        np.testing.assert_array_equal(dist_v(r1, r2, L), expected)

    def test_negative_and_positive(self):
        r1 = [-5, -5, -5]
        r2 = [5, 5, 5]
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(dist_v(r1, r2, L), expected)

if __name__ == '__main__':
    unittest.main()
