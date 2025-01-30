
import numpy as np
import unittest
from TOEG642PE2YV2K import WJ

class TestWeightedJacobiMethod(unittest.TestCase):
    def test_zero_initial_guess(self):
        A = np.array([[4, 1], [2, 3]])
        b = np.array([1, 2])
        eps = 1e-5
        x_true = np.linalg.solve(A, b)
        x0 = np.zeros(2)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

    def test_non_zero_initial_guess(self):
        A = np.array([[4, 1], [2, 3]])
        b = np.array([1, 2])
        eps = 1e-5
        x_true = np.linalg.solve(A, b)
        x0 = np.array([1, 1])
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

    def test_omega_zero(self):
        A = np.array([[3, 1], [1, 2]])
        b = np.array([5, 3])
        eps = 1e-5
        x_true = np.linalg.solve(A, b)
        x0 = np.zeros(2)
        omega = 0  # Omega zero should not change x from x0
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertNotAlmostEqual(residual, 0, delta=eps)
        self.assertNotAlmostEqual(error, 0, delta=eps)

    def test_large_matrix(self):
        A = np.diag(np.arange(1, 101))
        b = np.arange(1, 101)
        eps = 1e-5
        x_true = np.linalg.solve(A, b)
        x0 = np.zeros(100)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

    def test_incorrect_dimensions(self):
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1])  # Incorrect dimension
        eps = 1e-5
        x_true = np.array([1, -1])  # This is just a placeholder
        x0 = np.zeros(2)
        omega = 2/3
        with self.assertRaises(ValueError):
            residual, error = WJ(A, b, eps, x_true, x0, omega)

    def test_identity_matrix(self):
        A = np.eye(3)
        b = np.array([1, 2, 3])
        eps = 1e-5
        x_true = np.array([1, 2, 3])
        x0 = np.zeros(3)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

if __name__ == '__main__':
    unittest.main()
