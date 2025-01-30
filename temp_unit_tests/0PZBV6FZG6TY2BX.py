
import unittest
import numpy as np
from ZSHDPNDKC798S9R import Simulate

class TestSimulate(unittest.TestCase):
    def test_zero_initial_species(self):
        spc_init = np.zeros(2)
        res_init = np.array([100, 100])
        b = np.array([0.1, 0.1])
        c = np.array([[0.01, 0.02], [0.02, 0.01]])
        w = np.array([1, 1])
        m = np.array([0.5, 0.5])
        r = np.array([0.05, 0.05])
        K = np.array([100, 100])
        tf = 50
        dt = 0.1
        SPC_THRES = 10
        expected = []
        result = Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES)
        self.assertEqual(result, expected)

    def test_zero_initial_resources(self):
        spc_init = np.array([50, 50])
        res_init = np.zeros(2)
        b = np.array([0.1, 0.1])
        c = np.array([[0.01, 0.02], [0.02, 0.01]])
        w = np.array([1, 1])
        m = np.array([0.5, 0.5])
        r = np.array([0.05, 0.05])
        K = np.array([100, 100])
        tf = 50
        dt = 0.1
        SPC_THRES = 10
        expected = []
        result = Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES)
        self.assertEqual(result, expected)

    def test_high_maintenance_cost(self):
        spc_init = np.array([100, 100])
        res_init = np.array([100, 100])
        b = np.array([0.1, 0.1])
        c = np.array([[0.01, 0.02], [0.02, 0.01]])
        w = np.array([1, 1])
        m = np.array([100, 100])  # High maintenance cost
        r = np.array([0.05, 0.05])
        K = np.array([100, 100])
        tf = 50
        dt = 0.1
        SPC_THRES = 10
        expected = []
        result = Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES)
        self.assertEqual(result, expected)

    def test_negative_timestep(self):
        spc_init = np.array([50, 50])
        res_init = np.array([100, 100])
        b = np.array([0.1, 0.1])
        c = np.array([[0.01, 0.02], [0.02, 0.01]])
        w = np.array([1, 1])
        m = np.array([0.5, 0.5])
        r = np.array([0.05, 0.05])
        K = np.array([100, 100])
        tf = 50
        dt = -0.1  # Negative timestep
        SPC_THRES = 10
        with self.assertRaises(ValueError):
            Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES)

    def test_large_resource_carrying_capacity(self):
        spc_init = np.array([100, 100])
        res_init = np.array([100, 100])
        b = np.array([0.1, 0.1])
        c = np.array([[0.01, 0.02], [0.02, 0.01]])
        w = np.array([1, 1])
        m = np.array([0.5, 0.5])
        r = np.array([0.05, 0.05])
        K = np.array([10000, 10000])  # Large carrying capacity
        tf = 50
        dt = 0.1
        SPC_THRES = 10
        expected = [0, 1]
        result = Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
