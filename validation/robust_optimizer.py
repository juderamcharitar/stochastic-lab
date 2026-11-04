import pandas as pd,numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import t as student_t
from src.robustness_v09 import get_breaks,hfilter,load_returns
def repair(training,asymmetric,max_fit_obs=100000,maxiter=160):
 tr=training.iloc[-max_fit_obs:];raw=tr.to_numpy(float);mu=raw.mean();scale=raw.std(ddof=1);z=(raw-mu)/scale;breaks=get_breaks(tr.index)
 def objective(x):
  a,b=x[:2];g=x[2] if asymmetric else 0.
  if a+b+g/2>=.994:return 1e5
  h=hfilter(z,a,b,g,breaks)
  return float(np.mean(.5*(np.log(h)+z*z/h)))
 initials=[[.065,.87,.05],[.12,.75,.05],[.03,.90,.05],[.2,.65,.02]] if asymmetric else [[.065,.87],[.12,.75],[.03,.90],[.2,.65]]
 bounds=[(.002,.30),(.02,.98)]+([(.0,.32)] if asymmetric else [])
 candidates=[]
 for x0 in initials:
  opt=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':(lambda x: .994-x[0]-x[1]-(x[2]/2 if asymmetric else 0))}],options={'maxiter':maxiter,'ftol':1e-9})
  if opt.success and np.isfinite(opt.fun) and opt.fun<1e3 and opt.x[0]+opt.x[1]+(opt.x[2]/2 if asymmetric else 0)<.99401:
   candidates.append(opt)
 if not candidates:raise RuntimeError('All optimization starts failed')
 opt=min(candidates,key=lambda v:v.fun);a,b=map(float,opt.x[:2]);g=float(opt.x[2]) if asymmetric else 0.0
 resid=z/np.sqrt(hfilter(z,a,b,g,breaks))
 def t_loss(df):
  sf=np.sqrt((df-2)/df)
  return -float(np.mean(student_t.logpdf(resid/sf,df)-np.log(sf)))
 dof=float(minimize_scalar(t_loss,bounds=(2.2,50),method='bounded',options={'xatol':.1}).x)
 return dict(mu=mu,scale=scale,alpha=a,beta=b,gamma=g,df=dof,converged=True,n_fit=len(tr),objective=float(opt.fun))
