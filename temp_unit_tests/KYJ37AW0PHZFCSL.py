
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
        r = np.array([1.0, 2.0, 3.0])
        L = 10.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_exactly_on_edge(self):
        r = np.array([10.0, 10.0, 10.0])
        L = 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_outside_bounds_positive(self):
        r = np.array([11.0, 12.0, 13.0])
        L = 10.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_outside_bounds_negative(self):
        r = np.array([-1.0, -2.0, -3.0])
        L = 10.0
        expected = np.array([9.0, 8.0, 7.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([5.0, 15.0, -5.0])
        L = 0.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_large_values(self):
        r = np.array([1000.0, 2000.0, 3000.0])
        L = 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_array_input(self):
        r = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
        L = 10.0
        expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

if __name__ == '__main__':
    unittest.main()
