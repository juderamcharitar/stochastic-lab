import unittest
import numpy as np
import pandas as pd
from src.validation_v08 import joint_score,kupiec,independence_test
from src.monte_carlo_v08 import estimate,true_tail
from src.asymmetric_v08 import hfilter

class V08Tests(unittest.TestCase):
    def test_joint_score_finite(self):
        s=joint_score(np.array([0,.04]),np.array([.02,.02]),np.array([.03,.03]));self.assertTrue(np.isfinite(s).all())
    def test_es_positive_required(self):
        with self.assertRaises(ValueError):joint_score([.04],[.02],[0.])
    def test_kupiec_perfect(self):
        self.assertAlmostEqual(kupiec(np.array([1]*10+[0]*990))[0],0.,places=8)
    def test_independence_gaps(self):
        t=pd.to_datetime(['2025-01-01 00:00Z','2025-01-01 00:05Z','2025-01-01 00:10Z','2025-01-01 01:00Z','2025-01-01 01:05Z']*4,utc=True)
        # repeated timestamps are irrelevant to test transition counting
        _,_,n=independence_test(np.zeros(20),t)
        self.assertTrue(n<19)
    def test_gjr_nonnegative(self):
        h=hfilter(np.array([0,-2,1,-1.]),.05,.8,.1);self.assertTrue(np.all(h>0))
    def test_gjr_asymmetry(self):
        self.assertGreater(hfilter([0,-2,0],.05,.8,.1)[2],hfilter([0,2,0],.05,.8,.1)[2])
    def test_gaussian_truth(self):
        v,e=true_tail();self.assertGreater(e,v)
    def test_importance_finite(self):
        self.assertTrue(np.isfinite(estimate(2000,42,'importance')))
    def test_reproducible_mc(self):
        self.assertEqual(estimate(300,123,'importance'),estimate(300,123,'importance'))
if __name__=='__main__':unittest.main()
