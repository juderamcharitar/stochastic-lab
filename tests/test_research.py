import sys,unittest,tempfile,zipfile
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from research import timestamp_utc,read_zip_minutes,contiguous_returns,gaussian_var_es,bootstrap_paths,garch_fit,garch_forecast_vol,kupiec_pvalue
class Tests(unittest.TestCase):
 def test_ms_us(self):
  self.assertEqual(timestamp_utc(pd.Series([1788998400000])).iloc[0],timestamp_utc(pd.Series([1788998400000000])).iloc[0])
 def test_gap(self):
  ix=pd.DatetimeIndex(['2026-01-01 00:00Z','2026-01-01 00:05Z','2026-01-01 00:10Z','2026-01-02 00:00Z','2026-01-02 00:05Z'])
  r=contiguous_returns(pd.Series([1,2,4,8,16],index=ix)) if False else None
  s=pd.Series(np.arange(1,201,dtype=float),index=pd.date_range('2026-01-01',periods=200,freq='5min',tz='UTC'))
  joined=pd.concat([s,pd.Series([1000],index=[pd.Timestamp('2026-01-03',tz='UTC')])]);r=contiguous_returns(joined)
  self.assertEqual(len(r),199)
 def test_var(self):
  v,e=gaussian_var_es(0,.01);self.assertGreater(e,v);self.assertGreater(v,0)
 def test_bootstrap(self):
  x=np.arange(30)*.01;arr=bootstrap_paths(x,13,12,4,np.random.default_rng(1));self.assertEqual(arr.shape,(12,13))
 def test_garch(self):
  x=np.random.default_rng(3).normal(0,.01,300);m=garch_fit(x);vol=garch_forecast_vol(x,x[:10],m);self.assertTrue(np.all(vol>0))
 def test_kupiec(self):
  self.assertGreaterEqual(kupiec_pvalue(np.zeros(200,dtype=bool),.01),0)
 def test_archive(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'BTCUSDT-trades-2026-09-10.zip'
   with zipfile.ZipFile(path,'w') as z:z.writestr('test.csv','1,75000,0.1,7500,1788998400000000,False,True\n2,75001,0.1,7500,1788998401000000,False,True\n')
   x=read_zip_minutes(path);self.assertEqual(len(x),1);self.assertEqual(x.iloc[0],75001)
if __name__=='__main__':unittest.main()
