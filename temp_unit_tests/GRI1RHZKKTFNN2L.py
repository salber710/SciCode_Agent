
import numpy as np
import unittest
from testingCode import wrap

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

    def test_above_upper_bound(self):
        r = np.array([11, 12, 13])
        L = 10
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([0, 0, 0])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_negative(self):
        r = np.array([-20, -30, -40])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_positive(self):
        r = np.array([25, 35, 45])
        L = 10
        expected = np.array([5, 5, 5])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_non_integer_values(self):
        r = np.array([10.7, 10.5, 10.3])
        L = 10
        expected = np.array([0.7, 0.5, 0.3])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_negative_non_integer_values(self):
        r = np.array([-0.7, -0.5, -0.3])
        L = 10
        expected = np.array([9.3, 9.5, 9.7])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_zero_box_length(self):
        r = np.array([1, 2, 3])
        L = 0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_negative_length(self):
        r = np.array([1, 2, 3])
        L = -10
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_non_cubic_input(self):
        r = np.array([1, 2])
        L = 10
        with self.assertRaises(ValueError):
            wrap(r, L)

if __name__ == '__main__':
    unittest.main()
