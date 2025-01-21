
import numpy as np
import unittest
from testingCode import davidson_solver

class TestDavidsonSolver(unittest.TestCase):
    def test_small_symmetric_matrix(self):
        matrixA = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        num_eigenvalues = 2
        threshold = 1e-5
        expected_eigenvalues = np.sort(np.linalg.eigvalsh(matrixA)[:num_eigenvalues])
        result_eigenvalues = np.sort(davidson_solver(matrixA, num_eigenvalues, threshold))
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues, decimal=5)

    def test_identity_matrix(self):
        matrixA = np.eye(5)
        num_eigenvalues = 3
        threshold = 1e-5
        expected_eigenvalues = np.zeros(num_eigenvalues)
        result_eigenvalues = davidson_solver(matrixA, num_eigenvalues, threshold)
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues, decimal=5)

    def test_non_symmetric_matrix(self):
        matrixA = np.array([[1, 2], [3, 4]])
        num_eigenvalues = 1
        threshold = 1e-5
        with self.assertRaises(ValueError):
            davidson_solver(matrixA, num_eigenvalues, threshold)

    def test_large_symmetric_matrix(self):
        matrixA = np.diag(np.arange(1, 101))
        num_eigenvalues = 5
        threshold = 1e-5
        expected_eigenvalues = np.arange(1, num_eigenvalues + 1)
        result_eigenvalues = davidson_solver(matrixA, num_eigenvalues, threshold)
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues, decimal=5)

    def test_threshold_effect(self):
        matrixA = np.array([[10, 2], [2, 1]])
        num_eigenvalues = 1
        threshold_high = 1e-1
        threshold_low = 1e-5
        eigenvalues_high = davidson_solver(matrixA, num_eigenvalues, threshold_high)
        eigenvalues_low = davidson_solver(matrixA, num_eigenvalues, threshold_low)
        self.assertNotEqual(eigenvalues_high[0], eigenvalues_low[0])

    def test_zero_matrix(self):
        matrixA = np.zeros((4, 4))
        num_eigenvalues = 2
        threshold = 1e-5
        expected_eigenvalues = np.zeros(num_eigenvalues)
        result_eigenvalues = davidson_solver(matrixA, num_eigenvalues, threshold)
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues, decimal=5)

    def test_negative_eigenvalues(self):
        matrixA = np.array([[-2, 0], [0, -1]])
        num_eigenvalues = 2
        threshold = 1e-5
        expected_eigenvalues = np.sort(np.linalg.eigvalsh(matrixA))
        result_eigenvalues = np.sort(davidson_solver(matrixA, num_eigenvalues, threshold))
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues, decimal=5)

if __name__ == '__main__':
    unittest.main()
