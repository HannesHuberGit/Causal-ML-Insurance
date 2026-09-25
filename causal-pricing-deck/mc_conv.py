import numpy as np, json, sys
from conv import slearner, dml
from cont import draw
ns=[2000,5000,10000,20000,50000,100000,200000]; R=int(sys.argv[1])
out={}
for n in ns:
    rows=[]
    for r in range(R):
        A,T,Y,p=draw(n,5000+r); s,_=slearner(A,T,Y); th,se=dml(A,T,Y); rows.append([s,th,se])
    out[n]=rows; json.dump(out,open("out/mc_conv.json","w")); a=np.array(rows)
    print(n, a[:,0].mean().round(3), a[:,0].std().round(3), a[:,1].mean().round(3), a[:,1].std().round(3), flush=True)
