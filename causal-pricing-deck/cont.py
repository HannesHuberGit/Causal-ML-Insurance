import numpy as np, lightgbm as lgb, warnings, sys
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
THETA, SIG_V = 1.0, 0.02
def g(A): return 0.35 - 0.006*(A-25) + 0.015*np.sin((A-25)/6)
def m(A): return 0.0025*(A-25)            # continuous tariff: +0.25% per year of age
def draw(n, seed):
    r=np.random.default_rng(seed); A=r.uniform(25,75,n); V=r.normal(0,SIG_V,n); T=m(A)+V
    p=np.clip(g(A)+THETA*T,0,1); Y=r.binomial(1,p).astype(float); return A,T,Y,p
def P(rounds, leaves=31, mcs=50, lr=0.05): return dict(n_estimators=rounds, learning_rate=lr, num_leaves=leaves, min_child_samples=mcs, verbose=-1)
def slearner(A,T,Y,pp):
    mod=lgb.LGBMRegressor(**pp).fit(np.c_[A,T],Y); d=np.log1p(.02)-np.log1p(-.02)
    return (mod.predict(np.c_[A,T+np.log1p(.02)]).mean()-mod.predict(np.c_[A,T+np.log1p(-.02)]).mean())/d
def dml(A,T,Y,pp,k=5):
    X=A.reshape(-1,1); lh,mh=np.zeros_like(Y),np.zeros_like(T)
    for tr,te in KFold(k,shuffle=True,random_state=0).split(X):
        lh[te]=lgb.LGBMRegressor(**pp).fit(X[tr],Y[tr]).predict(X[te]); mh[te]=lgb.LGBMRegressor(**pp).fit(X[tr],T[tr]).predict(X[te])
    Yr,Tr=Y-lh,T-mh; th=(Tr@Yr)/(Tr@Tr); e=Yr-th*Tr; se=np.sqrt(np.mean(Tr**2*e**2))/np.mean(Tr**2)/np.sqrt(len(Y)); return th,se
if __name__=="__main__":
    A,T,Y,p=draw(200_000,7)
    print("pooled",np.polyfit(T,Y,1)[0], "corr",np.corrcoef(A,T)[0,1], "kappa", m(A).var()/SIG_V**2)
    for a in [30,40,50,60,70]:
        s=abs(A-a)<2; print(" slice",a,round(np.polyfit(T[s],Y[s],1)[0],3), s.sum())
    for R in [10,25,50,100,200,400,800]:
        print(R, "S", round(slearner(A,T,Y,P(R)),3), "DML", [round(x,3) for x in dml(A,T,Y,P(R))])
