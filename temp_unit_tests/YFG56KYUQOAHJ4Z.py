
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
    # Use numpy's remainder function to wrap the coordinates within the interval [0, L)
    coord = np.remainder(r, L)
    return coord

class TestWrapFunction(unittest.TestCase):
    def test_within_bounds(self):
        r = np.array([1.0, 2.0, 3.0])
        L = 10.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_on_upper_bound(self):
        r = np.array([10.0, 10.0, 10.0])
        L = 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_negative_coordinates(self):
        r = np.array([-1.0, -2.0, -3.0])
        L = 10.0
        expected = np.array([9.0, 8.0, 7.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_large_coordinates(self):
        r = np.array([11.0, 22.0, 33.0])
        L = 10.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([1.0, 2.0, 3.0])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_non_cubic_input(self):
        r = np.array([1.0, 2.0])
        L = 10.0
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_non_numeric_input(self):
        r = np.array(['a', 'b', 'c'])
        L = 10.0
        with self.assertRaises(TypeError):
            wrap(r, L)

    def test_large_and_negative_coordinates(self):
        r = np.array([-11.0, 22.0, -33.0])
        L = 10.0
        expected = np.array([9.0, 2.0, 7.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

if __name__ == '__main__':
    unittest.main()
