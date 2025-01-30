
import unittest
from ZBRK2TY8H0ET5ES import Bspline
import numpy as np

class TestNewBsplineFunction(unittest.TestCase):
    def test_degree_zero_exact_knot(self):
        Xi = np.array([0, 1, 2, 3])
        self.assertEqual(Bspline(0, 0, 0, Xi), 1.0)
        self.assertEqual(Bspline(1, 1, 0, Xi), 1.0)

    def test_degree_zero_boundary(self):
        Xi = np.array([0, 1, 2, 3])
        self.assertEqual(Bspline(1, 0, 0, Xi), 0.0)
        self.assertEqual(Bspline(2, 1, 0, Xi), 0.0)

    def test_degree_one_exact_knot(self):
        Xi = np.array([0, 1, 2, 3])
        self.assertAlmostEqual(Bspline(1, 0, 1, Xi), 0.0)
        self.assertAlmostEqual(Bspline(2, 1, 1, Xi), 0.0)

    def test_degree_two_values(self):
        Xi = np.array([0, 1, 2, 3, 4])
        self.assertAlmostEqual(Bspline(1.5, 0, 2, Xi), 0.125)
        self.assertAlmostEqual(Bspline(2.5, 1, 2, Xi), 0.625)

    def test_invalid_knot_vector(self):
        Xi = np.array([0])
        with self.assertRaises(ValueError):
            Bspline(0.5, 0, 0, Xi)

    def test_index_out_of_bounds(self):
        Xi = np.array([0, 1, 2, 3])
        with self.assertRaises(IndexError):
            Bspline(0.5, 3, 0, Xi)

    def test_degree_zero_negative_xi(self):
        Xi = np.array([-2, -1, 0, 1])
        self.assertEqual(Bspline(-1.5, 0, 0, Xi), 1.0)
        self.assertEqual(Bspline(-0.5, 1, 0, Xi), 1.0)

    def test_high_degree_non_uniform_knots(self):
        Xi = np.array([0, 0, 0, 1, 2, 3, 3, 3])
        self.assertAlmostEqual(Bspline(1.5, 2, 3, Xi), 0.375)

    def test_degree_with_repeated_knots(self):
        Xi = np.array([0, 1, 1, 2, 3])
        self.assertAlmostEqual(Bspline(1, 1, 1, Xi), 0.0)

    def test_large_degree(self):
        Xi = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertAlmostEqual(Bspline(5.5, 4, 5, Xi), 0.0)

if __name__ == '__main__':
    unittest.main()
