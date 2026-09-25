import numpy as np, warnings
from sklearn.model_selection import KFold
from cont import draw, g, m, THETA
from conv import fit_es
warnings.filterwarnings("ignore")
res={}
for n,R in [(2000,10),(5000,10),(20000,6),(50000,4),(200000,3)]:
    vals=[]
    for r in range(R):
        A,T,Y,p=draw(n,5000+r); X=A.reshape(-1,1); lh=np.zeros(n); mh=np.zeros(n)
        for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
            lh[te]=fit_es(X[tr],Y[tr]).predict(X[te]); mh[te]=fit_es(X[tr],T[tr]).predict(X[te])
        dm=mh-m(A); dl=lh-(g(A)+THETA*m(A)); dgg=dl-THETA*dm; v2=np.mean((T-m(A))**2)
        vals.append(np.mean(dm*dgg)/v2)
    res[n]=np.mean(vals); print(n, round(res[n],4), flush=True)
ns=np.array(list(res)); b=np.abs(np.array(list(res.values())))
print("DML product-term rate", np.polyfit(np.log(ns),np.log(b),1)[0].round(2))
