import numpy as np, lightgbm as lgb, warnings, sys, json, time
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings("ignore")
from cont import draw, THETA
BASE=dict(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1)
def fit_es(X,y,seed=0):
    Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=0.2,random_state=seed)
    mod=lgb.LGBMRegressor(**BASE).fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
    return mod
def slearner(A,T,Y):
    mod=fit_es(np.c_[A,T],Y); d=np.log1p(.02)-np.log1p(-.02)
    return (mod.predict(np.c_[A,T+np.log1p(.02)]).mean()-mod.predict(np.c_[A,T+np.log1p(-.02)]).mean())/d, mod.best_iteration_
def dml(A,T,Y,k=5):
    X=A.reshape(-1,1); lh,mh=np.zeros_like(Y),np.zeros_like(T)
    for tr,te in KFold(k,shuffle=True,random_state=0).split(X):
        lh[te]=fit_es(X[tr],Y[tr]).predict(X[te]); mh[te]=fit_es(X[tr],T[tr]).predict(X[te])
    Yr,Tr=Y-lh,T-mh; th=(Tr@Yr)/(Tr@Tr); e=Yr-th*Tr; se=np.sqrt(np.mean(Tr**2*e**2))/np.mean(Tr**2)/np.sqrt(len(Y)); return th,se
if __name__=="__main__":
  ns=[int(x) for x in sys.argv[1].split(",")]; R=int(sys.argv[2])
  for n in ns:
      t=time.time(); S=[];D=[];C=[];it=[]
      for r in range(R):
          A,T,Y,p=draw(n,1000+r); s,bi=slearner(A,T,Y); th,se=dml(A,T,Y); S.append(s); D.append(th); C.append(abs(th-THETA)<1.96*se); it.append(bi)
      S,D=np.array(S),np.array(D)
      print(n, f"S mean {S.mean():.3f} sd {S.std():.3f} | DML mean {D.mean():.3f} sd {D.std():.3f} cover {np.mean(C):.2f} | rounds {np.mean(it):.0f} | {time.time()-t:.0f}s", flush=True)
