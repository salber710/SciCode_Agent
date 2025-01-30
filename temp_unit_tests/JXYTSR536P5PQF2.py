
import numpy as np
import unittest
from N4LD3T8Z2MT7P6W import davidson_solver

class TestDavidsonSolverNew(unittest.TestCase):
    def test_small_symmetric_matrix(self):
        matrixA = np.array([[2, 1], [1, 2]])
        num_eigenvalues = 1
        threshold = 1e-6
        expected_eigenvalues = np.array([1])  # Smallest eigenvalue
        result = davidson_solver(matrixA, num_eigenvalues, threshold)
        np.testing.assert_allclose(result, expected_eigenvalues, atol=threshold)

    def test_non_square_matrix(self):
        matrixA = np.array([[1, 2, 3], [4, 5, 6]])  # Non-square matrix
        num_eigenvalues = 1
        threshold = 1e-6
        with self.assertRaises(ValueError):
            davidson_solver(matrixA, num_eigenvalues, threshold)

    def test_negative_threshold(self):
        matrixA = np.eye(3)
        num_eigenvalues = 2
        threshold = -0.01  # Negative threshold
        with self.assertRaises(ValueError):
            davidson_solver(matrixA, num_eigenvalues, threshold)

    def test_zero_eigenvalues_requested(self):
        matrixA = np.eye(3)
        num_eigenvalues = 0  # Requesting zero eigenvalues
        threshold = 1e-6
        with self.assertRaises(ValueError):
            davidson_solver(matrixA, num_eigenvalues, threshold)

    def test_more_eigenvalues_than_dimensions(self):
        matrixA = np.eye(3)
        num_eigenvalues = 4  # More eigenvalues requested than dimensions
        threshold = 1e-6
        with self.assertRaises(ValueError):
            davidson_solver(matrixA, num_eigenvalues, threshold)

    def test_random_symmetric_matrix(self):
        np.random.seed(0)
        matrixA = np.random.rand(10, 10)
        matrixA = (matrixA + matrixA.T) / 2  # Make it symmetric
        num_eigenvalues = 3
        threshold = 1e-6
        result = davidson_solver(matrixA, num_eigenvalues, threshold)
        expected_eigenvalues = np.sort(np.linalg.eigvalsh(matrixA))[:num_eigenvalues]
        np.testing.assert_allclose(result, expected_eigenvalues, atol=threshold)

    def test_matrix_with_zeros(self):
        matrixA = np.zeros((5, 5))
        num_eigenvalues = 3
        threshold = 1e-6
        expected_eigenvalues = np.zeros(num_eigenvalues)
        result = davidson_solver(matrixA, num_eigenvalues, threshold)
        np.testing.assert_allclose(result, expected_eigenvalues, atol=threshold)

if __name__ == '__main__':
    unittest.main()
