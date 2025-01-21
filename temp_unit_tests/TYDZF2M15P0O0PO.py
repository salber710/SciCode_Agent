
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

    def test_exactly_on_upper_bound(self):
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
        r = np.array([20.0, 30.0, 40.0])
        L = 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_zero_length_box(self):
        r = np.array([5.0, 15.0, 25.0])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_fractional_coordinates(self):
        r = np.array([10.5, 20.5, 30.5])
        L = 10.0
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_negative_box_size(self):
        r = np.array([1.0, 2.0, 3.0])
        L = -10.0
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_non_numeric_input(self):
        with self.assertRaises(TypeError):
            r = "not a numeric array"
            L = 10.0
            wrap(r, L)

if __name__ == '__main__':
    unittest.main()
