
import numpy as np
import unittest

# Function to be tested
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
        r = np.array([5, 5, 5])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([5, 5, 5]))

    def test_on_upper_bound(self):
        r = np.array([10, 10, 10])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))

    def test_negative_coordinates(self):
        r = np.array([-1, -1, -1])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([9, 9, 9]))

    def test_zero_length(self):
        r = np.array([0, 0, 0])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))

    def test_large_values(self):
        r = np.array([20, 30, 40])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))

    def test_fractional_values(self):
        r = np.array([9.5, 10.5, 11.5])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([9.5, 0.5, 1.5]))

    def test_large_negative_values(self):
        r = np.array([-20, -30, -40])
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))

    def test_zero_box_size(self):
        r = np.array([1, 2, 3])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_non_numeric_input(self):
        r = np.array(['a', 'b', 'c'])
        L = 10
        with self.assertRaises(TypeError):
            wrap(r, L)

    def test_non_array_input(self):
        r = [1, 2, 3]  # Not a numpy array
        L = 10
        result = wrap(r, L)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

# Running the tests
if __name__ == '__main__':
    unittest.main()
