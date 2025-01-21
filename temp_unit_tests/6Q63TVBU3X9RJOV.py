
import numpy as np
import unittest
from testingCode import wrap

class TestWrapFunction(unittest.TestCase):
    def test_basic_wrapping(self):
        r = np.array([10, 20, 30])
        L = 15
        expected = np.array([10, 5, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_negative_coordinates(self):
        r = np.array([-1, -2, -3])
        L = 10
        expected = np.array([9, 8, 7])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_zero_coordinates(self):
        r = np.array([0, 0, 0])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_exact_boundary(self):
        r = np.array([10, 20, 30])
        L = 10
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_negative_coordinates(self):
        r = np.array([-11, -22, -33])
        L = 10
        expected = np.array([9, 8, 7])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_large_positive_coordinates(self):
        r = np.array([101, 202, 303])
        L = 100
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_float_coordinates(self):
        r = np.array([10.75, 20.5, 30.25])
        L = 15
        expected = np.array([10.75, 5.5, 0.25])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_float_negative_coordinates(self):
        r = np.array([-1.5, -2.75, -3.25])
        L = 10
        expected = np.array([8.5, 7.25, 6.75])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_zero_length(self):
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
        r = [10, 20, 30]  # Not a numpy array
        L = 15
        expected = np.array([10, 5, 0])
        np.testing.assert_array_equal(wrap(np.array(r), L), expected)

    def test_non_integer_length(self):
        r = np.array([12.5, 15.5, 18.5])
        L = 10.5
        expected = np.array([2, 5, 8])
        np.testing.assert_array_equal(wrap(r, L), expected)

    def test_negative_length(self):
        r = np.array([1, 2, 3])
        L = -10
        with self.assertRaises(ValueError):
            wrap(r, L)

if __name__ == '__main__':
    unittest.main()
