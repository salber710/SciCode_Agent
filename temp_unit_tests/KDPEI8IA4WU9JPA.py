
import numpy as np
import unittest
from V0IOWG832LHPNBC import WJ

class TestWeightedJacobiMethod(unittest.TestCase):
    def test_convergence(self):
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        eps = 1e-5
        x_true = np.array([0.09090909, 0.63636364])
        x0 = np.zeros(2)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

    def test_non_convergence(self):
        A = np.array([[1, 2], [3, 4]])
        b = np.array([5, 11])
        eps = 1e-5
        x_true = np.array([1, 2])
        x0 = np.zeros(2)
        omega = 1.5  # Improper omega, should not converge
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertNotAlmostEqual(residual, 0, delta=eps)
        self.assertNotAlmostEqual(error, 0, delta=eps)

    def test_zero_matrix(self):
        A = np.zeros((2, 2))
        b = np.array([0, 0])
        eps = 1e-5
        x_true = np.zeros(2)
        x0 = np.zeros(2)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

    def test_identity_matrix(self):
        A = np.eye(2)
        b = np.array([1, 2])
        eps = 1e-5
        x_true = np.array([1, 2])
        x0 = np.zeros(2)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

    def test_large_matrix(self):
        A = np.diag(np.arange(1, 101))
        b = np.arange(1, 101)
        eps = 1e-5
        x_true = np.ones(100)
        x0 = np.zeros(100)
        omega = 2/3
        residual, error = WJ(A, b, eps, x_true, x0, omega)
        self.assertAlmostEqual(residual, 0, delta=eps)
        self.assertAlmostEqual(error, 0, delta=eps)

if __name__ == '__main__':
    unittest.main()
