import unittest
import numpy as np
import pandas as pd
from src.walkforward import segment_by_day, garch_fit_segmented,one_day_garch_variances,fit_t_df,quantile_score,run_walkforward

class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(11)
        dates=[]; vals=[]
        for d in range(8):
            ix=pd.date_range(pd.Timestamp('2026-09-01',tz='UTC')+pd.Timedelta(days=d*2),periods=60,freq='5min')
            dates.extend(ix);vals.extend(rng.normal(0,.002,60))
        self.returns=pd.Series(vals,index=pd.DatetimeIndex(dates))
    def test_no_join_across_days(self):
        segments=segment_by_day(self.returns)
        self.assertEqual(len(segments),8)
        self.assertTrue(all(len(s)==60 for s in segments))
    def test_garch_daily_reset(self):
        fit={'mu':0.,'scale':.01,'alpha':.1,'beta':.8,'omega':.1}
        v=one_day_garch_variances(np.array([.5,0.]),fit)
        self.assertAlmostEqual(v[0],.0001)
        self.assertGreater(v[1],v[0])
        self.assertAlmostEqual(one_day_garch_variances(np.array([0.]),fit)[0],.0001)
    def test_q_loss_nonnegative(self):
        q=quantile_score(np.array([-.2,0.,.3]),np.array([-.1,-.1,-.1]))
        self.assertTrue(np.all(q>=0))
    def test_student_t_valid(self):
        df=fit_t_df(np.random.default_rng(4).standard_t(7,size=300))
        self.assertGreater(df,2)
    def test_walkforward_chronological(self):
        d,s,f=run_walkforward(self.returns,min_train_days=5)
        self.assertEqual(f.shape[0],3)
        self.assertEqual(set(d.date),set(map(str,sorted(set(self.returns.index.date))[5:])))
        self.assertEqual(len(d),3*60*4)
        self.assertTrue(np.isfinite(s.mean_pinball).all())
    def test_insufficient_days(self):
        with self.assertRaises(ValueError):run_walkforward(self.returns,min_train_days=7)

if __name__=='__main__':unittest.main()
