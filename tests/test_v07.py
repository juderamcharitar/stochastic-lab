import tempfile,unittest,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.multiyear import load_month,load_history,time_unit
from src.tailrisk_v07 import normal_thresholds,t_thresholds,garch_variances,rolling_forecasts,validate_es
from src.monte_carlo_v07 import bs_call,convergence,estimate_call

class MultiyearTests(unittest.TestCase):
    def make_month(self,folder,year,month,unit='ms',drop=None):
        name=f'BTCUSDT-5m-{year}-{month:02d}.zip';p=Path(folder)/name
        start=pd.Timestamp(year,month,1,tz='UTC');stop=start+pd.offsets.MonthBegin(1)
        idx=pd.date_range(start,stop-pd.Timedelta(minutes=5),freq='5min')
        if drop is not None:idx=idx.delete(drop)
        stamps=idx.asi8//(1000 if unit=='us' else 10**6)
        csv=''.join(f'{int(t)},1,1,1,{100+i*.001},1,1,1,1,1,1,0\n' for i,t in enumerate(stamps))
        with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr(name[:-4]+'.csv',csv)
        (Path(str(p)+'.CHECKSUM')).write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
        return p
    def test_time_units(self):
        self.assertEqual(time_unit([1700000000000]),'ms')
        self.assertEqual(time_unit([1735689600000000]),'us')
    def test_both_timestamp_units_and_continuity(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_month(d,2024,1)
            self.make_month(d,2024,2)
            r,c=load_history(d)
            self.assertTrue(c['continuous']);self.assertEqual(c['gap_count'],0)
            self.assertEqual(len(r), (31+29)*288-1)
    def test_2025_microseconds(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make_month(d,2025,1,'us')
            s,c=load_month(p)
            self.assertEqual(len(s),31*288);self.assertEqual(c['timestamp_unit'],'us')
    def test_missing_candle_detected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_month(d,2024,1,drop=23)
            with self.assertRaises(ValueError):load_history(d)
            r,c=load_history(d,strict=False)
            self.assertFalse(c['continuous']);self.assertEqual(c['gap_count'],1)
    def test_checksum_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make_month(d,2024,1)
            with p.open('ab') as f:f.write(b'x')
            with self.assertRaises(ValueError):load_month(p)

class TailRiskTests(unittest.TestCase):
    def test_normal_es_formula(self):
        v,e=normal_thresholds(0,.02)
        self.assertGreater(e,v)
        self.assertAlmostEqual(v,-.02*norm.ppf(.01),places=10)
    def test_t_es_greater_than_var(self):
        v,e=t_thresholds(0,.02,5)
        self.assertGreater(e,v)
    def test_garch_recursion_no_future(self):
        z=np.array([.2,-.4,3.,-.1])
        h=garch_variances(z,.1,.8)
        changed=z.copy();changed[2]=200.
        h2=garch_variances(changed,.1,.8)
        np.testing.assert_allclose(h[:3],h2[:3]);self.assertNotAlmostEqual(h[3],h2[3])
    def test_synthetic_walkforward(self):
        rng=np.random.default_rng(12)
        idx=pd.date_range('2024-01-01',periods=288*12,freq='5min',tz='UTC')
        r=pd.Series(rng.normal(0,.005,len(idx)),index=idx)
        detail,fits=rolling_forecasts(r,train_days=5,refit_days=3)
        self.assertEqual(detail.model.nunique(),4)
        self.assertTrue(np.isfinite(detail.ES99).all())
        self.assertTrue((detail.ES99>detail.VaR99).all())
        summary=validate_es(detail,draws=70)
        self.assertEqual(len(summary),4)
        self.assertTrue((summary.days==7).all())

class MonteCarloTests(unittest.TestCase):
    def test_analytic_call(self):
        self.assertAlmostEqual(bs_call(100,100,0,.2,1),7.9655674554,places=6)
    def test_unbiased_methods_converge_in_expectation(self):
        _,sm,truth=convergence(samples=(512,2048),replications=24,seed=6)
        self.assertTrue(np.isfinite(sm.rmse).all())
        self.assertTrue(sm[(sm.n_paths==2048)&(sm.method=='control')].rmse.iloc[0]<sm[(sm.n_paths==2048)&(sm.method=='iid')].rmse.iloc[0])

if __name__=='__main__':unittest.main()
