
import numpy as np
import unittest

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    coord = np.mod(r, L)
    return coord

class TestWrapFunction(unittest.TestCase):
    def test_within_bounds(self):
        r = np.array([5, 5, 5])
        L = 10
        expected = np.array([5, 5, 5])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_on_upper_bound(self):
        r = np.array([10, 10, 10])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_below_zero(self):
        r = np.array([-1, -1, -1])
        L = 10
        expected = np.array([9, 9, 9])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_exactly_zero(self):
        r = np.array([0, 0, 0])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_above_upper_bound(self):
        r = np.array([11, 12, 13])
        L = 10
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_negative(self):
        r = np.array([-11, -12, -13])
        L = 10
        expected = np.array([9, 8, 7])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_non_integer_values(self):
        r = np.array([10.5, 10.75, 10.25])
        L = 10
        expected = np.array([0.5, 0.75, 0.25])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_values(self):
        r = np.array([100, 200, 300])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_array_of_zeros(self):
        r = np.array([0, 0, 0])
        L = 0
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_negative_box_size(self):
        r = np.array([1, 2, 3])
        L = -10
        with self.assertRaises(ValueError):
            wrap(r, L)

if __name__ == '__main__':
    unittest.main()
