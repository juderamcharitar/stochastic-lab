import unittest
import numpy as np
import pandas as pd
from src.market_sim_v10 import daily_panels,generate,summarize_paths,quantile_score,crps_mc,evaluate

class TestV10(unittest.TestCase):
    def setUp(self):
        self.p={'mean':0.0,'std':.01,'student_df':6.,'jump_probability':.002,'jump_mean_z':0.,'jump_sd_z':4.,'symmetric':{'mu':0,'scale':.01,'df':6.,'alpha':.07,'beta':.89,'gamma':0},'asymmetric':{'mu':0,'scale':.01,'df':6.,'alpha':.06,'beta':.85,'gamma':.1}}
    def test_gaussian_seed(self):
        a=generate('Gaussian IID',self.p,50,12,seed=7)
        np.testing.assert_array_equal(a,generate('Gaussian IID',self.p,50,12,seed=7))
    def test_sim_dimensions(self):
        for model in ('Gaussian IID','Student-t IID','Student-t GARCH','Student-t GJR-GARCH','Heuristic jump-diffusion'):
            arr=generate(model,self.p,30,15,seed=8)
            self.assertEqual(arr.shape,(30,15));self.assertTrue(np.isfinite(arr).all())
    def test_bootstrap_no_synthetic(self):
        bank=np.arange(24.).reshape(6,4)
        out=generate('Historical day bootstrap',self.p,100,4,seed=8,block_source=bank)
        self.assertTrue(all(any(np.array_equal(x,y) for y in bank) for x in out))
    def test_break_day_rejected(self):
        ix=pd.date_range('2025-01-01 00:00',periods=8,freq='5min',tz='UTC')
        x=pd.Series(np.arange(8.),index=ix)
        self.assertEqual(len(daily_panels(x,8)[0]),1)
        self.assertEqual(len(daily_panels(x.drop(ix[4]),8)[0]),0)
    def test_drawdown(self):
        d=summarize_paths(np.array([[.1,-.2,.05]]))
        self.assertAlmostEqual(d['max_log_drawdown'][0],-.2)
    def test_pinball(self):
        self.assertAlmostEqual(quantile_score([0.],-.2,.01),.002)
    def test_crps_nonnegative(self):
        self.assertGreaterEqual(crps_mc(np.array([0,1]),np.array([0,1])),0.)
    def test_evaluation_rows(self):
        obs=np.random.default_rng(2).normal(0,.01,(35,10))
        sim=np.random.default_rng(1).normal(0,.01,(1000,10))
        row=evaluate(sim,obs,'test')
        self.assertEqual(row['historical_days'],35)
        self.assertEqual(row['scenario_paths'],1000)
    def test_invalid_paths(self):
        with self.assertRaises(ValueError):generate('Gaussian IID',self.p,-1,10)

if __name__=='__main__':unittest.main()
