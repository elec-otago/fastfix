import unittest

from .. import util

class TestUtil(unittest.TestCase):
    
    def test_phase(self):
        
        self.assertAlmostEqual(util.phase_delta2(0.999,0.001), 0.002)
        self.assertAlmostEqual(util.phase_delta2(0.99, 0.01), 0.02, 4)

    def test_gaussian_llh(self):
        
        self.assertAlmostEqual(util.gaussian_llh(0,0,2), -1.612, 3)
        self.assertAlmostEqual(util.gaussian_llh(0,0,1), -0.919, 3)

