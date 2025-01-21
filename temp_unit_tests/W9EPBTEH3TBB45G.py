
import numpy as np
import unittest

def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (array_like): The three-dimensional displacement vector between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials.
    '''
    r_mag = np.linalg.norm(r)
    if r_mag < rc:
        sr = sigma / r_mag
        sr6 = sr ** 6
        force_magnitude = 24 * epsilon * (2 * sr6 ** 2 - sr6) / r_mag ** 2
        force_vector = force_magnitude * (r / r_mag)
    else:
        force_vector = np.array([0.0, 0.0, 0.0])
    return force_vector

class TestLennardJonesForce(unittest.TestCase):
    def test_force_at_cutoff(self):
        r = np.array([rc, 0, 0])
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(f_ij(r, sigma, epsilon, rc), expected)

    def test_force_below_cutoff(self):
        r = np.array([0.5 * rc, 0, 0])
        result = f_ij(r, sigma, epsilon, rc)
        self.assertTrue(np.linalg.norm(result) > 0)
        np.testing.assert_array_almost_equal(result, -result / np.linalg.norm(result) * np.linalg.norm(r))

    def test_force_above_cutoff(self):
        r = np.array([1.5 * rc, 0, 0])
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(f_ij(r, sigma, epsilon, rc), expected)

    def test_zero_distance(self):
        r = np.array([0, 0, 0])
        with self.assertRaises(ZeroDivisionError):
            f_ij(r, sigma, epsilon, rc)

    def test_negative_epsilon(self):
        r = np.array([sigma, 0, 0])
        with self.assertRaises(ValueError):
            f_ij(r, sigma, -epsilon, rc)

    def test_negative_sigma(self):
        r = np.array([rc, 0, 0])
        with self.assertRaises(ValueError):
            f_ij(r, -sigma, epsilon, rc)

    def test_repulsive_force_close_range(self):
        r = np.array([0.1, 0, 0])
        result = f_ij(r, sigma, epsilon, rc)
        self.assertTrue(np.linalg.norm(result) > 0)
        self.assertTrue(result[0] > 0)

    def test_attractive_force_intermediate_range(self):
        r = np.array([0.8, 0, 0])
        result = f_ij(r, sigma, epsilon, rc)
        self.assertTrue(np.linalg.norm(result) > 0)
        self.assertTrue(result[0] < 0)

    def test_force_direction(self):
        r = np.array([1, 1, 0])
        force = f_ij(r, sigma, epsilon, rc)
        expected_direction = r / np.linalg.norm(r)
        calculated_direction = force / np.linalg.norm(force)
        np.testing.assert_array_almost_equal(calculated_direction, expected_direction)

    def test_force_magnitude(self):
        r = np.array([0.5, 0.5, 0.5])
        force = f_ij(r, sigma, epsilon, rc)
        self.assertTrue(np.linalg.norm(force) > 0)

# Constants used in the tests
sigma = 1.0
epsilon = 1.0
rc = 2.5

if __name__ == '__main__':
    unittest.main()
