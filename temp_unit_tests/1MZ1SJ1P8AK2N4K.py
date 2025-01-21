
import numpy as np
import unittest
from code import wrap

class TestWrapFunction(unittest.TestCase):
    def test_within_bounds(self):
        r = np.array([1.0, 2.0, 3.0])
        L = 10.0
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_on_boundary(self):
        r = np.array([10.0, 10.0, 10.0])
        L = 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_above_boundary(self):
        r = np.array([10.1, 20.2, 30.3])
        L = 10.0
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_negative_coordinates(self):
        r = np.array([-1.0, -2.0, -3.0])
        L = 10.0
        expected = np.array([9.0, 8.0, 7.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_zero_length(self):
        r = np.array([1.0, 2.0, 3.0])
        L = 0.0
        with self.assertRaises(ZeroDivisionError):
            wrap(r, L)

    def test_large_coordinates(self):
        r = np.array([1000.0, 2000.0, 3000.0])
        L = 10.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_small_box(self):
        r = np.array([0.5, 0.5, 0.5])
        L = 0.3
        expected = np.array([0.2, 0.2, 0.2])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_non_float_input(self):
        r = np.array([1, 2, 3])
        L = 10
        expected = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(wrap(r, L), expected)

    def test_non_numeric_input(self):
        r = np.array(['a', 'b', 'c'])
        L = 10
        with self.assertRaises(TypeError):
            wrap(r, L)

    def test_negative_length_box(self):
        r = np.array([1, 2, 3])
        L = -10
        with self.assertRaises(ValueError):
            wrap(r, L)

    def test_vector_length_not_three(self):
        r = np.array([1.5, 2.5])
        L = 10
        with self.assertRaises(ValueError):
            wrap(r, L)

if __name__ == '__main__':
    unittest.main()
