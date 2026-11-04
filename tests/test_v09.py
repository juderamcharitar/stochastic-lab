import unittest
import numpy as np
import pandas as pd
from src.robustness_v09 import get_breaks,hfilter,fit_model,thresholds,joint_score,paired_bootstrap,summarize
class Robustness09Tests(unittest.TestCase):
 def test_breaks(self):
  x=pd.to_datetime(['2024-01-01 23:50Z','2024-01-01 23:55Z','2024-01-02 00:00Z','2024-01-02 00:10Z'],utc=True)
  self.assertEqual(get_breaks(x).tolist(),[True,False,True,True])
 def test_causal(self):
  z=np.array([0.4,-2.,1.,2.]);b=np.array([True,False,False,False])
  p=hfilter(z,.1,.8,.05,b)
  zz=z.copy();zz[2]=123
  q=hfilter(zz,.1,.8,.05,b)
  self.assertTrue(np.allclose(p[:3],q[:3]))
  self.assertNotAlmostEqual(p[-1],q[-1])
 def test_reset(self):
  z=np.array([1.,100.,0.,0.]);b=np.array([True,False,True,False])
  h=hfilter(z,.05,.8,0,b)
  self.assertEqual(h[2],1.)
 def test_asym(self):
  z=np.array([0.,-2,1,0]);h=hfilter(z,.1,.75,.1,np.array([True,False,False,False]))
  self.assertGreater(h[2],1.)
 def test_positive_threshold(self):
  v,e=thresholds(0,np.array([.02,.03]),5)
  self.assertTrue(np.all(e>v))
 def test_score_requires_valid_es(self):
  with self.assertRaises(ValueError):joint_score(np.array([1]),np.array([.5]),np.array([0]))
 def test_bootstrap_equal_scores(self):
  dates=['2025-01-%02d'%i for i in range(1,11)]
  r=[]
  for d in dates:
   for name in ['Symmetric Student-t GARCH','Asymmetric Student-t GJR']:
    r.append(dict(date=d,train_days=60,model=name,n=10,joint_score_sum=-30))
  out=paired_bootstrap(pd.DataFrame(r),draws=50)
  self.assertAlmostEqual(out.iloc[0].asymmetric_minus_symmetric_joint_score,0)
 def test_fit(self):
  rng=np.random.default_rng(6);times=pd.date_range('2025-01-01',periods=1600,freq='5min',tz='UTC')
  r=pd.Series(rng.standard_t(7,1600)*.001,index=times)
  fit=fit_model(r,True,max_fit_obs=1200,maxiter=10)
  self.assertTrue(fit['alpha']+fit['beta']+.5*fit['gamma']<1)
  self.assertEqual(fit['n_fit'],1200)
if __name__=='__main__':unittest.main()
