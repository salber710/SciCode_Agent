
import numpy as np
import unittest
from code import wrap

class TestWrapFunction(unittest.TestCase):
    def test_within_bounds(self):
        r = np.array([5, 5, 5])
        L = 10
        expected = np.array([5, 5, 5])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_on_boundary(self):
        r = np.array([10, 10, 10])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_above_boundary(self):
        r = np.array([12, 15, 18])
        L = 10
        expected = np.array([2, 5, 8])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_negative_coordinates(self):
        r = np.array([-1, -5, -10])
        L = 10
        expected = np.array([9, 5, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([1, 2, 3])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_large_numbers(self):
        r = np.array([1000, 2000, 3000])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_small_numbers(self):
        r = np.array([0.1, 0.2, 0.3])
        L = 1
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_negative_length(self):
        r = np.array([1, 2, 3])
        L = -10
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_non_integer_values(self):
        r = np.array([1.5, 2.5, 3.5])
        L = 10
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_vector_length_not_three(self):
        r = np.array([1, 2])
        L = 10
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_fractional_values(self):
        r = np.array([10.75, -2.5, 9.25])
        L = 10
        expected = np.array([0.75, 7.5, 9.25])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_non_numeric_input(self):
        r = np.array(['a', 'b', 'c'])
        L = 10
        with self.assertRaises(TypeError):
            wrap(r, L)

    def test_non_array_input(self):
        r = [1.0, 2.0, 3.0]  # Not a numpy array
        L = 10
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(np.array(r), expected)

if __name__ == '__main__':
    unittest.main()
