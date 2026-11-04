import unittest
import numpy as np
import pandas as pd
from src.tailrisk_v07 import rolling_forecasts,fit_garch

class GapSafetyTests(unittest.TestCase):
    def test_walkforward_gaps(self):
        rng=np.random.default_rng(9)
        idx=pd.date_range('2024-01-01',periods=288*10,freq='5min',tz='UTC')
        idx=idx.delete([33, 34, 430, 1120, 1700])
        r=pd.Series(rng.normal(0,.004,len(idx)),index=idx)
        detail,_=rolling_forecasts(r,train_days=5,refit_days=3)
        self.assertEqual(detail.model.nunique(),4)
        self.assertTrue(np.isfinite(detail.ES99).all())
        self.assertTrue((detail.ES99>detail.VaR99).all())
    def test_segmented_fits_are_finite(self):
        rng=np.random.default_rng(6)
        r=rng.normal(0,.003,1000)
        params=fit_garch(r,[0,100,530])
        self.assertGreater(params['df'],2)
        self.assertLess(params['alpha']+params['beta'],1)

if __name__=='__main__':unittest.main()
