import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import numpy as np,pandas as pd
from robustness import build_surface,summarize_surface,create_figure,create_heatmap

class TestSurface(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(11)
        idx=pd.date_range('2026-01-01',periods=9*288,freq='5min',tz='UTC')
        self.r=pd.Series(rng.normal(0,.001,len(idx)),index=idx)
    def test_future_days_only(self):
        f=build_surface(self.r,train_days=4,alpha_grid=[.05,.1],beta_grid=[.8,.9])
        self.assertEqual(f.date.nunique(),5)
        self.assertTrue(all(f.train_end < f.date))
    def test_invalid_domain_mask(self):
        f=build_surface(self.r,train_days=4,alpha_grid=[.1],beta_grid=[.9])
        self.assertTrue(f.loss.isna().all())
    def test_gap_day_not_carried(self):
        r=self.r.copy();r=r[~(r.index.date==pd.Timestamp('2026-01-05').date())]
        f=build_surface(r,train_days=4,alpha_grid=[.05],beta_grid=[.8])
        self.assertEqual(f.date.nunique(),4)
    def test_summary_and_chart(self):
        f=build_surface(self.r,train_days=4,alpha_grid=[.04,.09],beta_grid=[.8,.85]);s=summarize_surface(f)
        self.assertEqual(len(s),5)
        self.assertEqual(len(create_figure(f).frames),5)
        self.assertEqual(len(create_heatmap(f).frames),5)
    def test_reproducible(self):
        f1=build_surface(self.r,train_days=4,alpha_grid=[.04],beta_grid=[.8]);f2=build_surface(self.r,train_days=4,alpha_grid=[.04],beta_grid=[.8]);pd.testing.assert_frame_equal(f1,f2)
    def test_bad_timestamps(self):
        with self.assertRaises(ValueError):build_surface(pd.Series([.1]*20,index=range(20)))
if __name__=='__main__':unittest.main()
