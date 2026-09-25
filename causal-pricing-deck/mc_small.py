import numpy as np, json
from conv import slearner, dml
from cont import draw
ns=[100,200,500,1000]; R=30
out=json.load(open("out/mc_conv.json"))
for n in ns:
    rows=[]
    for r in range(R):
        A,T,Y,p=draw(n,5000+r); s,_=slearner(A,T,Y); th,se=dml(A,T,Y); rows.append([s,th,se])
    out[str(n)]=rows; a=np.array(rows)
    print(n, "S", a[:,0].mean().round(3), a[:,0].std().round(3), np.percentile(a[:,0],[10,90]).round(2), "| DML", a[:,1].mean().round(3), a[:,1].std().round(3), np.percentile(a[:,1],[10,90]).round(2), flush=True)
json.dump(out,open("out/mc_conv.json","w"))
