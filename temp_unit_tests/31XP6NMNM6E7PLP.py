
import numpy as np
import unittest

def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the difference vector between r1 and r2
    delta = np.array(r1) - np.array(r2)
    
    # Apply minimum image convention to each component of the vector
    delta = delta - L * np.round(delta / L)
    
    # Calculate the Euclidean distance of the resulting vector
    distance = np.linalg.norm(delta)
    
    return distance

class TestMinimumImageDistance(unittest.TestCase):
    def test_basic(self):
        r1 = (1, 1, 1)
        r2 = (2, 2, 2)
        L = 10
        expected = np.sqrt(3)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_periodic_boundary(self):
        r1 = (9, 9, 9)
        r2 = (1, 1, 1)
        L = 10
        expected = np.sqrt(3)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_large_distance_within_box(self):
        r1 = (0, 0, 0)
        r2 = (5, 5, 5)
        L = 10
        expected = np.sqrt(75)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_zero_distance(self):
        r1 = (5, 5, 5)
        r2 = (5, 5, 5)
        L = 10
        expected = 0
        result = dist(r1, r2, L)
        self.assertEqual(result, expected)

    def test_negative_coordinates(self):
        r1 = (-1, -1, -1)
        r2 = (-2, -2, -2)
        L = 10
        expected = np.sqrt(3)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_large_L(self):
        r1 = (100, 100, 100)
        r2 = (110, 110, 110)
        L = 1000
        expected = np.sqrt(300)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_small_L(self):
        r1 = (0.1, 0.1, 0.1)
        r2 = (0.2, 0.2, 0.2)
        L = 0.5
        expected = np.sqrt(0.03)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_boundary_crossing(self):
        r1 = (0, 0, 0)
        r2 = (9.9, 9.9, 9.9)
        L = 10
        expected = np.sqrt(0.03)
        result = dist(r1, r2, L)
        self.assertAlmostEqual(result, expected, places=5)

    def test_large_numbers(self):
        r1 = np.array([1000, 1000, 1000])
        r2 = np.array([1001, 1001, 1001])
        L = 2000
        expected_distance = np.sqrt(3)
        self.assertAlmostEqual(dist(r1, r2, L), expected_distance)

    def test_fractional_coordinates(self):
        self.assertAlmostEqual(dist([0.5, 0.5, 0.5], [9.5, 9.5, 9.5], 10), np.sqrt(1**2 + 1**2 + 1**2))

    def test_negative_box_size(self):
        with self.assertRaises(ValueError):
            dist([1, 2, 3], [4, 5, 6], -10)

    def test_zero_box_size(self):
        with self.assertRaises(ValueError):
            dist([1, 2, 3], [4, 5, 6], 0)

if __name__ == '__main__':
    unittest.main()
