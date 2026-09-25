import numpy as np, json
from conv import slearner, dml
from cont import draw
out=json.load(open("out/mc_conv.json"))
for n in [300,400]:
    rows=[]
    for r in range(30):
        A,T,Y,p=draw(n,5000+r); s,_=slearner(A,T,Y); th,se=dml(A,T,Y); rows.append([s,th,se])
    out[str(n)]=rows; a=np.array(rows)
    print(n,"LGBM",a[:,0].mean().round(3),"DML mean",a[:,1].mean().round(3),"median",np.median(a[:,1]).round(3),"p10/p90",np.percentile(a[:,1],[10,90]).round(2),flush=True)
json.dump(out,open("out/mc_conv.json","w"))
