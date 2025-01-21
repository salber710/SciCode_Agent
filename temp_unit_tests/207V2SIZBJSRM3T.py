
import numpy as np
import unittest

# The wrap function to be tested
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

# Unit tests for the wrap function
class TestWrapFunction(unittest.TestCase):
    def test_within_bounds(self):
        r = np.array([1.0, 2.0, 3.0])
        L = 5.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_exactly_on_upper_bound(self):
        r = np.array([5.0, 5.0, 5.0])
        L = 5.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_negative_coordinates(self):
        r = np.array([-1.0, -2.0, -3.0])
        L = 5.0
        expected = np.array([4.0, 3.0, 2.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_large_values(self):
        r = np.array([10.0, 20.0, 30.0])
        L = 5.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([1.0, 2.0, 3.0])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_non_numeric_input(self):
        r = np.array(['a', 'b', 'c'])
        L = 5.0
        with self.assertRaises(TypeError):
            wrap(r, L)

    def test_non_array_input(self):
        r = [1.0, 2.0, 3.0]  # List instead of numpy array
        L = 5.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(np.array(r), L), expected)

    def test_fractional_values(self):
        r = np.array([5.5, 10.75, 15.25])
        L = 5.0
        expected = np.array([0.5, 0.75, 0.25])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

# Run the tests
if __name__ == '__main__':
    unittest.main()
