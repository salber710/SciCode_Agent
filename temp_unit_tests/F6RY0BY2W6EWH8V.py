
import numpy as np
import unittest
from code import wrap

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

    def test_below_lower_bound(self):
        r = np.array([-1, -2, -3])
        L = 10
        expected = np.array([9, 8, 7])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_above_upper_bound(self):
        r = np.array([11, 12, 13])
        L = 10
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_values(self):
        r = np.array([100, 200, 300])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_small_negative_values(self):
        r = np.array([-100, -200, -300])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_non_integer_values(self):
        r = np.array([10.5, 20.5, 30.5])
        L = 10
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_fractional_values(self):
        r = np.array([10.75, 20.5, 30.25])
        L = 10.0
        expected = np.array([0.75, 0.5, 0.25])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_large_numbers(self):
        r = np.array([1e10 + 1, 1e10 + 2, 1e10 + 3])
        L = 1e10
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([1, 2, 3])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_negative_length(self):
        r = np.array([1, 2, 3])
        L = -10
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_non_numeric_input(self):
        r = np.array(['a', 'b', 'c'])
        L = 10
        with self.assertRaises(TypeError):
            wrap(r, L)

if __name__ == '__main__':
    unittest.main()
