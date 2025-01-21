
import unittest

def E_ij(r, sigma, epsilon, rc):
    '''Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The combined potential energy between the two particles, considering the specified potentials.
    '''
    if r >= rc:
        return 0.0

    # Calculate the Lennard-Jones potential at distance r
    lj_potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    
    # Calculate the Lennard-Jones potential at the cutoff distance rc
    lj_potential_rc = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)

    # Truncate and shift the potential
    return lj_potential - lj_potential_rc

class TestLennardJonesPotential(unittest.TestCase):
    def test_basic_functionality(self):
        sigma = 1.0
        epsilon = 1.0
        rc = 2.5
        r = 1.0
        expected = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6) - 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        self.assertAlmostEqual(E_ij(r, sigma, epsilon, rc), expected)

    def test_cutoff_distance(self):
        sigma = 1.0
        epsilon = 1.0
        rc = 2.5
        self.assertEqual(E_ij(rc, sigma, epsilon, rc), 0.0)
        self.assertEqual(E_ij(rc + 0.1, sigma, epsilon, rc), 0.0)

    def test_negative_epsilon(self):
        sigma = 1.0
        epsilon = -1.0
        rc = 2.5
        r = 1.0
        expected = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6) - 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        self.assertAlmostEqual(E_ij(r, sigma, epsilon, rc), expected)

    def test_zero_epsilon(self):
        sigma = 1.0
        epsilon = 0.0
        rc = 2.5
        r = 1.0
        self.assertEqual(E_ij(r, sigma, epsilon, rc), 0.0)

    def test_zero_sigma(self):
        sigma = 0.0
        epsilon = 1.0
        rc = 2.5
        r = 1.0
        with self.assertRaises(ZeroDivisionError):
            E_ij(r, sigma, epsilon, rc)

    def test_negative_distance(self):
        sigma = 1.0
        epsilon = 1.0
        rc = 2.5
        r = -1.0
        with self.assertRaises(ValueError):
            E_ij(r, sigma, epsilon, rc)

    def test_zero_distance(self):
        sigma = 1.0
        epsilon = 1.0
        rc = 2.5
        r = 0.0
        with self.assertRaises(ZeroDivisionError):
            E_ij(r, sigma, epsilon, rc)

    def test_large_distance(self):
        sigma = 1.0
        epsilon = 1.0
        rc = 2.5
        r = 100.0
        self.assertEqual(E_ij(r, sigma, epsilon, rc), 0.0)

    def test_negative_sigma(self):
        with self.assertRaises(ValueError):
            E_ij(1, -1, 1, 2.5)

    def test_negative_rc(self):
        with self.assertRaises(ValueError):
            E_ij(1, 1, 1, -1)

if __name__ == '__main__':
    unittest.main()
